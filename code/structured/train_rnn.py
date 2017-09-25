
import os, sys
import re
import math
import cPickle as pickle
import shutil
import io
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import pandas as pd
import time
import datetime


from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.layers.crf import ChainCRF
from keras.layers.wrappers import TimeDistributed
from keras.layers import Masking
from keras.callbacks import EarlyStopping
from keras.layers.pooling import GlobalAveragePooling1D


from FableLite import W

from tools import compute_stats_multiclass, compute_stats_binary
from tools import umls_lookup



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
    #tasks = dev_outcomes.values()[0].keys()
    #tasks = ['hosp_expire_flag']
    #tasks = ['diagnosis']
    tasks = ['gender']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit'])
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

        Y = vectorize_Y(train_ids, train_Y)
        num_tags = Y.shape[1]

        # build model
        lstm_model = create_lstm_model(vectorizers, num_tags, X, Y)

        # fit model
        X_doc,X_dts = X
        lstm_model.fit([X_doc,X_dts], Y, epochs=5, verbose=1)

        model = (criteria, vectorizers, lstm_model)
        models[task] = model

    for task,model in models.items():
        criteria, vectorizers, lstm_model = model
        vect = vectorizers[0]

        # train data
        train_labels,_ = filter_task(train_outcomes, task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X,_ = vectorize_X(train_ids, train_text_features, vectorizers=vectorizers)
        train_Y = vectorize_Y(train_ids, train_labels)

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        dev_X,_ = vectorize_X(dev_ids, dev_text_features, vectorizers=vectorizers)
        dev_Y = vectorize_Y(dev_ids, dev_labels)

        with io.StringIO() as out_f:
            # analysis

            # eval on dev data
            results(model, train_ids, train_X, train_Y, hours, 'TRAIN', task, out_f)
            results(model,   dev_ids,   dev_X,   dev_Y, hours, 'DEV'  , task, out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, dev_ids, dev_notes, dev_text_features, dev_X, dev_Y, hours, 'DEV', task)

        # serialize trained model
        modelname = '../../models/structured/bow/%s_%s.model' % (mode,task)
        M = {'criteria':criteria, 'vect':vectorizers, 'model':lstm_pickle(lstm_model), 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def lstm_pickle(lstm):
    print 'TODO: pickle by saving weights and reading'
    exit()



def vectorize_Y(ids, y_dict):
    # extract labels into list
    labels = set(y_dict.values())
    num_tags = len(set(labels))
    Y = np.zeros((len(ids),num_tags))
    for i,pid in enumerate(ids):
        ind = y_dict[pid]
        Y[i,ind] = 1
    return Y



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
    emb_size = W['and'].shape[0]

    # compute features
    for pid,records in notes.items():
        features = extract_text_features(records, hours)
        features_list[pid] = features

    # no-op
    df = None

    return features_list, df



def vectorize_X(ids, text_features, vectorizers=None):

    '''
    ids = ids[:3]
    text_features = {k:v for k,v in text_features.items() if k in ids}
    '''

    if vectorizers is None:
        # need reasonable limit on number of timesteps
        num_docs = max(map(len,text_features.values()))
        if num_docs > 32:
            num_docs = 32

        vectorizers = (num_docs,)

    num_samples = len(ids)
    emb_size = W['and'].shape[0]
    num_docs = vectorizers[0]

    dts = np.zeros((num_samples,num_docs,1))
    X = np.zeros((num_samples,num_docs,emb_size))
    for i,pid in enumerate(ids):
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            # right padding
            dts[i,num_docs-j-1,0] = dt.seconds
            X[i,num_docs-j-1,:] = centroid

    return (X,dts), vectorizers



def create_lstm_model(vectorizers, num_tags, X_dts, Y):

    X,dts = X_dts

    num_docs = vectorizers[0]
    emb_size = X.shape[2]

    print X.shape
    print Y.shape

    # document w2v centroids
    X_input  = Input(shape=(num_docs,emb_size) , dtype='float32', name='doc')
    seq = Bidirectional(LSTM(128, return_sequences=True))(X_input)

    # delta t of timestamps
    dt_input = Input(shape=(num_docs,1), dtype='float32', name='dt')

    # combine inputs
    combined = Concatenate()([seq, dt_input])
    combined_seq = Bidirectional(LSTM(128))(combined)

    # Predict target
    pred = Dense(num_tags, activation='softmax')(combined_seq)

    # Putting it all together
    model = Model(inputs=[X_input,dt_input], outputs=pred)
    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model



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

    elif task in ['sapsii','age','los']:
        if per_task_criteria is None:
            scores = sorted([ y[task] for y in Y.values() ])
            # make quartiles
            N = len(scores)
            num_bins = 4
            thresholds = []
            for i in range(1,num_bins):
                frac = float(i)/num_bins
                threshold = scores[int(frac * N)]
                thresholds.append(threshold)
            thresholds.append(float('inf'))
            '''
            print thresholds
            bins = defaultdict(list)
            for s in scores:
                for i in range(num_bins):
                    if s < thresholds[i]:
                        bins[i].append(s)
                        break
            for bin_id,sbin in bins.items():
                print bin_id, len(sbin)
            #exit()
            '''

            # save the "good" diagnoses (to extract same set from dev)
            per_task_criteria = thresholds
        else:
            thresholds = per_task_criteria

        ids = Y.keys()

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
    if task in ['sapsii', 'age', 'los']:
        thresholds = per_task_criteria
        scores = { pid:y[task] for pid,y in Y.items() }
        Y = {}
        for pid,s in scores.items():
            for i,threshold in enumerate(thresholds):
                if s < threshold:
                    Y[pid] = i
                    break
        '''
        counts = defaultdict(int)
        for s in Y.values():
            counts[s] += 1
        print counts
        '''

    else:
        filtered_Y = {pid:y[task] for pid,y in Y.items() 
                              if normalize_y(y[task],task) in per_task_criteria}
        filtered_normed_Y = {pid:normalize_y(y, task) for pid,y in filtered_Y.items() 
                              if normalize_y(y,task)!='**ignore**'}
        Y = {pid:per_task_criteria[y] for pid,y in filtered_normed_Y.items()}
    return Y, per_task_criteria




def results(model, ids, X, onehot_Y, hours, label, task, out_f):
    criteria, vectorizers, lstm_model = model
    vect = vectorizers[0]

    # for AUC
    P = lstm_model.predict(list(X))

    train_pred = P.argmax(axis=1)
    Y = onehot_Y.argmax(axis=1)
    num_tags = len(set(Y))

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if num_tags == 2:
        scores = P[:,1] - P[:,0]
        compute_stats_binary(task, train_pred, scores, Y, criteria, out_f)
    else:
        compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))



def error_analysis(model, ids, notes, text_features, X, onehot_Y, hours, label, task):
    criteria, vectorizers, clf = model
    vect = vectorizers[0]

    if task in ['sapsii', 'age', 'los']:
        V = {}
        labels_ = [0] + criteria
        for i in range(len(labels_)-1):
            label = '[%d,%s)' % (labels_[i],labels_[i+1])
            V[i] = label
    else:
        V = {v:k for k,v in criteria.items()}
    V[len(V)] = '**wrong**'

    # for confidence
    P = clf.predict(list(X))
    pred = P.argmax(axis=1)
    Y = onehot_Y.argmax(axis=1)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    thisdir = os.path.dirname(os.path.abspath(__file__))
    taskdir = os.path.join(thisdir, 'output', task)
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)
    methoddir = os.path.join(taskdir, 'bow')
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
            print >>f, 'SCORES'
            pind = conf[2]
            rind = conf[3]
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
    # collect all words
    bow = defaultdict(int)
    for w in toks:
        bow[w] += 1
    # only return words that have CUIs
    cui_bow = defaultdict(int)
    for w,c in bow.items():
        for cui in umls_lookup.cui_lookup(w):
            cui_bow[cui] += c
    return cui_bow



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
    features = []
    for note in notes:
        dt = note[0]
        #print dt
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if note[0] < datetime.timedelta(days=hours/24.0):
            # access the note's info
            section = note[1]
            toks = tokenize(note[2])

            bow = make_bow(toks)

            # compute centroid now (i.e. keras cant fine-tune)
            emb_size = W['and'].shape[0]
            w2v_sum = np.zeros(emb_size)
            N = 0
            for w,v in bow.items():
                if w in W:
                    w2v_sum += v * W[w]
                    N += v
            w2v_centroid = w2v_sum / (N+1e-9)

            features.append( (dt,w2v_centroid) )
    return features



if __name__ == '__main__':
    main()