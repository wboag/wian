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
import gensim.models
Doc2Vec = gensim.models.doc2vec.Doc2Vec


from tools import compute_stats_multiclass



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '240'], hours_s
    hours = float(hours_s)

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    # put notes in form needed for testing/training doc2vec model
    train_tagged_docs = preprocess_notes(train_notes, hours)
    dev_tagged_docs   = preprocess_notes(  dev_notes, hours)

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['diagnosis']
    #tasks = ['gender']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit', 'sapsii','los','age'])
    for task in tasks:
        if task in excluded:
            continue

        # extract appropriate data
        train_Y, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_Y.keys())
        print 'task:', task
        print 'N:   ', len(train_Y)

        # extract labels into list
        Y = np.array([train_Y[pid] for pid in train_ids])
        assert sorted(set(Y)) == range(max(Y)+1), 'need one of each example'
        
        # extract features into list
        X = [train_tagged_docs[pid] for pid in train_ids]

        # Do doc2vec for feature dimension reduction
        doc2vec = doc2vec_features_fit(X)
        X_doc2vec = doc2vec_features_transform(doc2vec, X)

        # learn SVM
        clf = LinearSVC(C=1e1)
        clf.fit(X_doc2vec, Y)

        model = (criteria, clf, doc2vec)
        models[task] = model

    for task,model in models.items():
        criteria, clf, doc2vec = model

        # train data
        train_labels,_ = filter_task(train_outcomes, task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X = [train_tagged_docs[pid] for pid in train_ids]
        train_X_doc2vec = doc2vec_features_transform(doc2vec, train_X)
        train_Y = np.array([train_labels[pid] for pid in train_ids])

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        dev_X = [dev_tagged_docs[pid] for pid in dev_ids]
        dev_X_doc2vec = doc2vec_features_transform(doc2vec, dev_X)
        dev_Y = np.array([dev_labels[pid] for pid in dev_ids])

        with io.StringIO() as out_f:
            # analysis
            analyze(clf, criteria, out_f)

            # eval on dev data
            results(model,train_ids,train_X_doc2vec,train_Y,hours,'TRAIN',task,out_f)
            results(model,  dev_ids,  dev_X_doc2vec,  dev_Y,hours,'DEV'  ,task,out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, dev_ids, dev_notes, dev_tagged_docs, dev_X_doc2vec, dev_Y, hours, 'DEV', task)

        # serialize trained model
        modelname = '../../models/structured/doc2vec/%s_%s.model' % (mode,task)
        M = {'criteria':criteria, 'clf':clf, 'doc2vec':doc2vec, 
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


def preprocess_notes(notes, hours, tokens_only=False):
    preprocessed_notes = {}
    
    # dummy record (prevent SVM from collparsing to single-dimension pred)
    preprocessed_notes[-1] = gensim.models.doc2vec.TaggedDocument(['foo'], [0])
    
    # get concatted notes
    for pid,records in notes.items():
        concat_note = extract_concat_note(records, hours)
        if tokens_only:
            preprocessed_notes[pid] = concat_note  
        else:
	    tagged_doc = gensim.models.doc2vec.TaggedDocument(concat_note, [pid])
            preprocessed_notes[pid] = tagged_doc

    return preprocessed_notes


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



def analyze(clf, labels_map, out_f):

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
    criteria, clf, doc2vec = model

    # for AUC
    P = clf.decision_function(X)

    train_pred = clf.predict(X)

    assert all(map(int,P.argmax(axis=1)) == train_pred)

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))



def error_analysis(model, ids, notes, text_features, X, Y, hours, label, task):
    criteria, clf, doc2vec = model

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
    methoddir = os.path.join(taskdir, 'doc2vec')
    if not os.path.exists(taskdir):
        os.makedirs(taskdir)
    if os.path.exists(methoddir):
        shutil.rmtree(methoddir)
    os.mkdir(methoddir)

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
        if pid == -1: continue
        if conf[2] == conf[3]:
            success = ''
        else:
            success = '_'
        filename = os.path.join(methoddir, '%s%s.pred' % (success,pid))
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

def extract_concat_note(notes, hours):
    concat_note = []     
    for note in notes:
        dt = note[0]
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if dt < datetime.timedelta(days=hours/24.0):
            # access the note's info
            category = note[1]
            tokens = tokenize(note[2])
            concat_note += tokens
    return concat_note


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



def doc2vec_features_fit(X):
    model = Doc2Vec(size=50, min_count=2, iter=100)
    model.build_vocab(X)
    model.train(X, total_examples=model.corpus_count, epochs=model.iter)
    return model

def doc2vec_features_transform(model, X):
    X_words = [tagged_doc.words for tagged_doc in X]
    return [model.infer_vector(doc) for doc in X_words]
    

def load_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text


def dump_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)



if __name__ == '__main__':
    main()
