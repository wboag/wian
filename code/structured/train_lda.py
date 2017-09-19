
import os, sys
from os.path import dirname
import commands
import re
import math
import cPickle as pickle
import shutil
import random
import io
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import pandas as pd
import datetime
import tempfile


from tools import compute_stats_multiclass



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '240'], hours_s
    hours = float(hours_s)

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    # vectorize notes
    train_text_features, df  = extract_features_from_notes(train_notes, hours, df=None)
    dev_text_features  , df_ = extract_features_from_notes(  dev_notes, hours, df=df)
    assert df == df_

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['diagnosis']
    tasks = ['gender']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit', 'sapsii','los','age'])
    for task in tasks:
        if task in excluded:
            continue

        #task = 'admission_location'
        #task = 'diagnosis'

        # extract appropriate data
        train_Y, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_Y.keys())
        print 'task:', task
        print 'N:   ', len(train_Y)

        # vecotrize notes
        X, vectorizers = vectorize_X(train_ids, train_text_features, vectorizers=None)

        # extract labels into list
        Y = np.array([train_Y[pid] for pid in train_ids])
        assert sorted(set(Y)) == range(max(Y)+1), 'need one of each example'

        # Do LDA for feature dimension reduction
        lda = lda_features_fit(X)
        X_lda = lda_features_transform(lda, X)

        # learn SVM
        clf = LinearSVC(C=1e1)
        clf.fit(X_lda, Y)

        model = (criteria, vectorizers, clf, lda)
        models[task] = model

    for task,model in models.items():
        criteria, vectorizers, clf, lda = model
        vect = vectorizers[0]

        # train data
        train_labels,_ = filter_task(train_outcomes, task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X,_ = vectorize_X(train_ids,train_text_features,vectorizers=vectorizers)
        train_X_lda = lda_features_transform(lda, train_X)
        train_Y = np.array([train_labels[pid] for pid in train_ids])

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        dev_X,_ = vectorize_X(dev_ids,dev_text_features,vectorizers=vectorizers)
        dev_X_lda = lda_features_transform(lda, dev_X)
        dev_Y = np.array([dev_labels[pid] for pid in dev_ids])

        with io.StringIO() as out_f:
            # analysis
            analyze(vect, clf, criteria, out_f)

            # eval on dev data
            results(model,train_ids,train_X_lda,train_Y,hours,'TRAIN',task,out_f)
            results(model,  dev_ids,  dev_X_lda,  dev_Y,hours,'DEV'  ,task,out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, dev_ids, dev_notes, dev_text_features, dev_X_lda, dev_Y, hours, 'DEV', task)

        # serialize trained model
        modelname = '../../models/structured/bow/%s_%s.model' % (mode,task)
        M = {'criteria':criteria, 'vect':vectorizers, 'clf':clf, 'lda':lda, 
             'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def get_data(datatype, mode):
    filename = '../../data/structured_%s_%s.pickle' % (mode,datatype)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        outcomes = pickle.load(f)

    assert sorted(X.keys()) == sorted(outcomes.keys())
    return X, outcomes



def extract_features_from_notes(notes, hours, df=None):
    features_list = {}

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    features_list[-1] = {'foo':1}

    # compute features
    for pid,records in notes.items():
        features = extract_text_features(records, hours)
        features_list[pid] = features

    # build doc freqs if we need them for training
    if df is None:
        df = defaultdict(int)
        for feats in features_list.values():
            for w in feats.keys():
                df[w] += 1

    return features_list, df



def vectorize_X(ids, text_features, vectorizers=None):
    # learn vectorizer on training data
    if vectorizers is None:
        vect = DictVectorizer()
        vect.fit(text_features.values())
        vectorizers = (vect,)

    filtered_features = [ text_features[pid] for pid in ids ]

    vect = vectorizers[0]
    X = vect.transform(filtered_features)

    return X, vectorizers



def filter_task(Y, task, per_task_criteria=None):

    # If it's a diagnosis, then only include diagnoses that occur >= 10 times
    if task == 'diagnosis':
        if per_task_criteria is None:
            # count diagnosis frequency
            counts = defaultdict(int)
            for y in Y.values():
                counts[y[task]] += 1

            '''
            for y,c in sorted(counts.items(), key=lambda t:t[1]):
                print '%4d %s' % (c,y)
            exit()
            '''

            # only include diagnois that are frequent enough
            diagnoses = {}
            top5 = sorted(counts.values())[-5:][0]
            for y,count in counts.items():
                #if count >= 350:
                if count >= top5:
                    diagnoses[y] = len(diagnoses)

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = diagnoses
        else:
            diagnoses = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (y[task] in diagnoses)]

    elif task == 'gender':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                counts[y[task]] += 1

            # only include diagnois that are frequent enough
            genders = {gender:i for i,gender in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = genders
        else:
            genders = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (y[task] in genders)]

    elif task == 'insurance':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                counts[y[task]] += 1

            # only include diagnois that are frequent enough
            genders = {gender:i for i,gender in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = genders
        else:
            genders = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (y[task] in genders)]

    elif task == 'ethnicity':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'language':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'marital_status':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'admission_location':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'discharge_location':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'hosp_expire_flag':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (normalize_y(y[task],task) in races)]

    elif task == 'admission_type':
        if per_task_criteria is None:
            counts = defaultdict(int)
            for y in Y.values():
                normed = y[task]
                counts[normed] += 1

            # only include diagnois that are frequent enough
            races = {race:i for i,race in enumerate(counts.keys())}

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = races
        else:
            races = per_task_criteria

        # which patients have that diagnosis?
        ids = [pid for pid,y in Y.items() if (y[task] in races)]

    else:
        print task
        counts = defaultdict(int)
        for y in Y.values():
            counts[y[task]] += 1
        hist = defaultdict(int)
        for y,count in counts.items():
            hist[count] += 1

        for y,count in sorted(counts.items(), key=lambda t:t[1]):
            print '%5d %s' % (count,y)

        print 'beep bop boop'
        exit()

    # return filtered data
    filtered_Y = {pid:y[task] for pid,y in Y.items() 
                          if normalize_y(y[task],task) in per_task_criteria}
    filtered_normed_Y = {pid:normalize_y(y, task) for pid,y in filtered_Y.items() 
                          if normalize_y(y,task)!='**ignore**'}
    Y = {pid:per_task_criteria[y] for pid,y in filtered_normed_Y.items()}
    Y[-1] = len(per_task_criteria)
    return Y, per_task_criteria



def analyze(vect, clf, labels_map, out_f):

    ind2feat =  { i:f for f,i in vect.vocabulary_.items() }
    labels = [label for label,i in sorted(labels_map.items(), key=lambda t:t[1])]

    # most informative features
    #"""
    out_f = sys.stdout
    out_f.write(unicode(clf.coef_.shape))
    out_f.write(unicode('\n\n'))
    num_feats = 10
    informative_feats = np.argsort(clf.coef_)

    for i,label in enumerate(labels):

        neg_features = informative_feats[i,:num_feats ]
        pos_features = informative_feats[i,-num_feats:]

        #'''
        # display what each feature is
        out_f.write(unicode('POS %s\n' % label))
        for feat in reversed(pos_features):
            val = clf.coef_[i,feat]
            word = feat
            if val > 1e-4:
                out_f.write(unicode('\t%-25s: %7.4f\n' % (word,val)))
            else:
                break
        out_f.write(unicode('NEG %s\n' % label))
        for feat in reversed(neg_features):
            val = clf.coef_[i,feat]
            word = feat
            if -val > 1e-4:
                out_f.write(unicode('\t%-25s: %7.4f\n' % (word,val)))
            else:
                break
        out_f.write(unicode('\n\n'))
        #'''
        #exit()
        #"""



def results(model, ids, X, Y, hours, label, task, out_f):
    criteria, vectorizers, clf, lda = model
    vect = vectorizers[0]

    # for AUC
    P = clf.decision_function(X)

    train_pred = clf.predict(X)

    assert all(map(int,P.argmax(axis=1)) == train_pred)

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))



def error_analysis(model, ids, notes, text_features, X, Y, hours, label, task):
    criteria, vectorizers, clf, lda = model
    vect = vectorizers[0]

    V = {v:k for k,v in criteria.items()}
    V[len(V)] = '**wrong**'

    # for confidence
    P = clf.decision_function(X)
    pred = clf.predict(X)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    thisdir = os.path.dirname(os.path.abspath(__file__))
    taskdir = os.path.join(thisdir, 'output', task)
    if os.path.exists(taskdir):
        shutil.rmtree(taskdir)
    os.mkdir(taskdir)

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
        if pid == -1: continue
        if conf[2] == conf[3]:
            success = ''
        else:
            success = '_'
        filename = os.path.join(taskdir, '%s%s.pred' % (success,pid))
        with open(filename, 'w') as f:
            print >>f, ''
            print >>f, '=' * 80
            print >>f, ''
            print >>f, pid
            print >>f, 'scores:', conf[1]
            print >>f, 'pred:  ', V[conf[2]]
            print >>f, 'ref:   ', V[conf[3]]
            print >>f, '#'*20
            print >>f, '#'*20
            for dt,category,text in sorted(notes[pid]):
                print >>f, dt
                print >>f, category
                print >>f, text
                print >>f, '-'*50
            print >>f, ''
            print >>f, '+'*80
            print >>f, ''



def softmax(scores):
    e = np.exp(scores)
    return e / e.sum()



def make_bow(toks):
    bow = defaultdict(int)
    for w in toks:
        bow[w] += 1
    return bow



def tokenize(text):
    text = text.lower()
    text = re.sub('[\',\.\-/\n]', ' ', text)
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = re.sub('\d[\d ]+', ' 0 ', text)
    return text.split()



def normalize_y(label, task):
    if task == 'ethnicity':
        if 'BLACK' in label: return 'BLACK'
        if 'ASIAN' in label: return 'ASIAN'
        if 'WHITE' in label: return 'WHITE'
        if 'LATINO' in label: return 'HISPANIC'
        if 'HISPANIC' in label: return 'HISPANIC'
        if 'SOUTH AMERICAN' in label: return 'HISPANIC'
        if 'PACIFIC' in label: return 'ASIAN'
        if 'PORTUGUESE' in label: return 'WHITE'
        if 'CARIBBEAN' in label: return 'HISPANIC'
        if 'AMERICAN INDIAN' in label: return '**ignore**'
        if 'ALASKA NATIVE' in label: return '**ignore**'
        if 'MIDDLE EASTERN' in label: return '**ignore**'
        if 'MULTI RACE' in label: return '**ignore**'
        if 'UNKNOWN' in label: return '**ignore**'
        if 'UNABLE TO OBTAIN' in label: return '**ignore**'
        if 'PATIENT DECLINED TO ANSWER' in label: return '**ignore**'
        if 'OTHER' in label: return '**ignore**'
    elif task == 'language':
        if label == 'ENGL': return 'ENGL'
        if label == 'SPAN': return 'SPAN'
        if label == 'RUSS': return 'RUSS'
        if label == 'PTUN': return 'PTUN'
        if label == 'CANT': return 'CANT'
        if label == 'PORT': return 'PORT'
        return '**ignore**'
    elif task == 'marital_status':
        if label == 'MARRIED': return 'MARRIED'
        if label == 'SINGLE': return 'SINGLE'
        if label == 'WIDOWED': return 'WIDOWED'
        if label == 'DIVORCED': return 'DIVORCED'
        return '**ignore**'
    elif task == 'admission_location':
        if label == 'EMERGENCY ROOM ADMIT': return 'EMERGENCY ROOM ADMIT'
        if label == 'PHYS REFERRAL/NORMAL DELI': return 'PHYS REFERRAL/NORMAL DELI'
        if label == 'TRANSFER FROM HOSP/EXTRAM': return 'TRANSFER FROM HOSP/EXTRAM'
        if label == 'CLINIC REFERRAL/PREMATURE': return 'CLINIC REFERRAL/PREMATURE'
        return '**ignore**'
    elif task == 'discharge_location':
        if 'HOME' in label: return 'HOME'
        if label == 'SNF': return 'SNF'
        if label == 'REHAB/DISTINCT PART HOSP': return 'REHAB/DISTINCT PART HOSP'
        if label == 'DEAD/EXPIRED': return 'DEAD/EXPIRED'
        return '**ignore**'
    return label



def extract_text_features(notes, hours):
    features = defaultdict(int)
    features['b'] = 1.0

    for note in notes:
        dt = note[0]
        #print dt
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if note[0] < datetime.timedelta(days=hours/24.0):
            # access the note's info
            section = note[1]
            toks = tokenize(note[2])

            bow = make_bow(toks)

            #'''
            # unigram features
            for w,tf in bow.items():
                featname = w
                '''
                if section:
                    featname = ('unigram', w, section)
                else:
                    featname = ('unigram', w)
                '''
                features[featname] += tf
                #features[featname] += 1
                #features[featname] = 1
            #'''

            '''
            # ngram features
            for n in [1,2]:
                for i in range(len(toks)-n+1):
                    ngram = tuple(toks[i:i+n])
                    if section:
                        featname = ('%d-gram'%n, ngram, section)
                    else:
                        featname = ('%d-gram'%n, ngram)
                    features[featname] = 1.0
            '''

    return dict(features)




def lda_features_fit(X):
    '''
    lda_model = '/tmp/wboag-lda-552519'
    lda = load_lda_model(lda_model)
    return lda
    '''

    lda_model = '/tmp/wboag-lda-%d' % random.randint(0,1000000)
    print lda_model

    # where the LDA executable is located
    home_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    ld = os.path.join(home_dir, 'resources', 'lda-c-dist')

    # Dump all data to file in LDA input format
    _, lda_file = tempfile.mkstemp(prefix='wboag-', dir='/tmp')
    with open(lda_file, 'w') as f:
        for i,x in enumerate(X):
            M = len(x.nonzero()[0])
            print >>f, M, 
            for ind,val in enumerate(x.todense().tolist()[0]):
                if val > 0:
                    print >>f, '%d:%d' % (ind, val), 
            print >>f, ''

    # shell out to LDA to train model
    cmd = '%s/lda est 0.1 50 %s/settings.txt %s random %s' % (ld,ld,lda_file,lda_model)
    status,output = commands.getstatusoutput(cmd)
    assert status==0, output

    # load the LDA model to keep it for pickle-ing
    lda = load_lda_model(lda_model)

    # clean up after yourself
    os.remove(lda_file)
    shutil.rmtree(lda_model)

    return lda



def lda_features_transform(lda, X):
    # Where LDA will output the proportions
    out = '/tmp/wboag-lda-%d' % random.randint(0,1000000)

    # where the LDA executable is located
    home_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    ld = os.path.join(home_dir, 'resources', 'lda-c-dist')

    # Dump all data to file in LDA input format
    _, lda_file = tempfile.mkstemp(prefix='wboag-', dir='/tmp')
    with open(lda_file, 'w') as f:
        for i,x in enumerate(X):
            M = len(x.nonzero()[0])
            print >>f, M, 
            for ind,val in enumerate(x.todense().tolist()[0]):
                if val > 0:
                    print >>f, '%d:%d' % (ind, val), 
            print >>f, ''

    # dump LDA model to file 
    lda_model = '/tmp/wboag-lda-%d' % random.randint(0,1000000)
    os.mkdir(lda_model)
    dump_lda_model(lda_model, lda)

    # shell out to LDA to train model
    cmd = '%s/lda inf %s/settings.txt %s/final %s %s'%(ld,ld,lda_model,lda_file,out)
    status,output = commands.getstatusoutput(cmd)
    assert status==0, output

    # load topic proportions
    topic_proportions = []
    with open('%s-gamma.dat'%out, 'r') as f:
        for line in f.readlines():
            vec = map(float,line.strip().split())
            topic_proportions.append(vec)
    X_lda = np.array(topic_proportions)

    # clean up after yourself
    os.remove(lda_file)
    shutil.rmtree(lda_model)

    return X_lda




def load_lda_model(lda_model):
    beta  = load_file('%s/final.beta'  % lda_model)
    gamma = load_file('%s/final.gamma' % lda_model)
    other = load_file('%s/final.other' % lda_model)
    return beta, gamma, other


def dump_lda_model(lda_model, lda):
     beta, gamma, other = lda
     dump_file('%s/final.beta'  % lda_model,  beta)
     dump_file('%s/final.gamma' % lda_model, gamma)
     dump_file('%s/final.other' % lda_model, other)


def load_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text


def dump_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)



if __name__ == '__main__':
    main()
