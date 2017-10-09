
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

#from FableLite import W

from tools import compute_stats_multiclass, compute_stats_binary
from tools import get_data, build_df, make_bow, filter_task, load_word2vec
from tools import get_treatments



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '2400'], hours_s
    hours = float(hours_s)

    N = int(sys.argv[3])

    train_notes, train_outcomes = get_data('train', mode)
    print 'loaded train notes'
    dev_notes  ,   dev_outcomes = get_data('dev' , mode)
    print 'loaded dev notes'

    train_ids = train_notes.keys()
    dev_ids   =   dev_notes.keys()

    T = get_treatments(mode)

    def select(d, ids):
        return {pid:val for pid,val in d.items() if pid in ids}
    train_treatments = {ttype:select(treats,train_ids) for ttype,treats in T.items()}
    dev_treatments   = {ttype:select(treats,  dev_ids) for ttype,treats in T.items()}


    # vectorize notes
    train_text_features, df  = extract_features_from_notes(train_notes, hours, N,df=None)
    print 'extracted train features'
    dev_text_features  , df_ = extract_features_from_notes(  dev_notes, hours, N,df=df)
    print 'extracted dev features'
    assert df == df_

    '''
    with open('embeddings_features.pickle', 'wb') as f:
        pickle.dump(train_text_features, f)
    exit()
    '''

    # throw out any data points that do not have any notes
    good_train_ids = [pid for pid,feats in train_text_features.items() if feats]
    good_dev_ids   = [pid for pid,feats in   dev_text_features.items() if feats]

    def filter_ids(items, ids):
        return {k:v for k,v in items.items() if k in ids}

    train_text_features = filter_ids(train_text_features, good_train_ids)
    train_outcomes      = filter_ids(train_outcomes     , good_train_ids)
    train_treatments = {task:filter_ids(its,good_train_ids) for task,its in train_treatments.items()}

    dev_text_features   = filter_ids(  dev_text_features,  good_dev_ids)
    dev_outcomes      = filter_ids(dev_outcomes     , good_dev_ids)
    dev_treatments = {task:filter_ids(its,good_dev_ids) for task,its in dev_treatments.items()}

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['los']
    #tasks = ['hosp_expire_flag']
    #tasks = ['diagnosis']
    #tasks = ['gender']
    tasks = ['ethnicity']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit', 'language', 'marital_status', 'insurance', 'discharge_location', 'admission_location'])

    for task in tasks:
        if task in excluded:
            continue

        #task = 'admission_location'
        #task = 'diagnosis'

        # extract appropriate data
        train_Y, criteria = filter_task(train_outcomes, train_treatments,
                                        task, per_task_criteria=None)
        train_ids = sorted(train_Y.keys())
        print 'task:', task
        print 'N:   ', len(train_Y)

        # vecotrize notes
        X,vectorizers = vectorize_X(train_ids, train_text_features, vectorizers=None)

        Y = vectorize_Y(train_ids, train_Y, criteria)

        # learn SVM
        clf = LinearSVC(C=1e-1)
        clf.fit(X, Y)

        model = (criteria, vectorizers, clf)
        models[task] = model

    for task,model in models.items():
        criteria, vectorizers, clf = model
        vect = vectorizers[0]

        # train data
        train_labels,_ = filter_task(train_outcomes, train_treatments,
                                     task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X,_ = vectorize_X(train_ids, train_text_features, vectorizers=vectorizers)
        train_Y = vectorize_Y(train_ids, train_labels, criteria)

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, dev_treatments,
                                   task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        #assert sorted(dev_ids) == sorted(dev_text_features)
        dev_X,_ = vectorize_X(dev_ids, dev_text_features, vectorizers=vectorizers)
        dev_Y = vectorize_Y(dev_ids, dev_labels, criteria)

        with io.StringIO() as out_f:
            # analysis

            # eval on dev data
            results(model,train_ids,train_X,train_Y,hours,'TRAIN',task,criteria,out_f)
            results(model,  dev_ids,  dev_X,  dev_Y,hours,'DEV'  ,task,criteria,out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, dev_ids, dev_notes, dev_text_features, dev_X, dev_Y, hours, 'DEV', task)

        # serialize trained model
        modelname = '../../models/structured/embeddings/%s_%s.model' % (mode,task)
        M = {'criteria':criteria, 'vect':vectorizers, 'clf':clf, 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def extract_features_from_notes(notes, hours, N, df=None):
    features_list = {}

    # no-op
    if df is None:
        df = build_df(notes, hours)

    # compute features
    for pid,records in sorted(notes.items()):
        features = extract_text_features(records, hours, N, df)
        features_list[pid] = features

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    dimensions = features_list.values()[0][0][1].shape[0]
    features_list[-1] = [(datetime.timedelta(days=0),np.zeros(dimensions))]

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

    #assert sorted(ids) == sorted(text_features.keys())
    dimensions = text_features.values()[0][0][1].shape[0]
    doc_embeddings = defaultdict(list)
    for i,pid in enumerate(ids):
        assert len(text_features[pid])>0, pid
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            doc_embeddings[pid].append(centroid)
    doc_embeddings = dict(doc_embeddings)
    #assert sorted(ids) == sorted(doc_embeddings.keys())

    # agrregate document centroids
    dimensions = text_features.values()[0][0][1].shape[0]
    X = np.zeros((len(ids),3*dimensions))
    for i,pid in enumerate(ids):
        vecs = doc_embeddings[pid]

        tmp = np.array(vecs)

        assert len(vecs)>0, pid
        min_vec = tmp.min(axis=0)
        max_vec = tmp.max(axis=0)
        avg_vec = tmp.sum(axis=0) / float(len(vecs))

        pid_vector = np.concatenate([min_vec,max_vec,avg_vec])

        X[i,:] = pid_vector

    #exit()

    return X, vectorizers



def vectorize_Y(ids, y_dict, criteria):
    # extract labels into list
    labels = set(y_dict.values())
    num_tags = len(criteria)+1
    Y = np.zeros(len(ids), dtype='int32')
    for i,pid in enumerate(ids):
        ind = y_dict[pid]
        Y[i] = ind
    return Y



def results(model, ids, X, Y, hours, label, task, labels, out_f):
    criteria, vectorizers, clf = model
    vect = vectorizers[0]

    # for AUC
    P = clf.decision_function(X)[:,:-1]
    train_pred = P.argmax(axis=1)

    # what is the predicted vocab without the dummy label?
    if task in ['los','age','sapsii']:
        V = range(len(labels))
    else:
        V = labels.keys()

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if len(V) == 2:
        scores = P[1:,1] - P[1:,0]
        compute_stats_binary(task, train_pred[1:], scores, Y[1:], criteria, out_f)
    else:
        compute_stats_multiclass(task,train_pred[1:],P[1:,:],Y[1:],criteria,out_f)
    out_f.write(unicode('\n\n'))



def error_analysis(model, ids, notes, text_features, X, Y, hours, label, task):
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
    P = clf.decision_function(X)
    pred = P.argmax(axis=1)

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



def extract_text_features(notes, hours, N, df):
    features = []
    for note in notes[:24]:
        dt = note[0]
        #print dt
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if note[0] < datetime.timedelta(days=hours/24.0):
            # access the note's info
            section = note[1]
            toks = note[2]

            bow = make_bow(toks)

            # select top-20 words by tfidf
            tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
            tfidf_in = {k:v for k,v in tfidf.items() if k in W}
            topN = sorted(tfidf_in.items(), key=lambda t:t[1])[-N:]

            if len(topN) < 1:
                continue

            vecs = [ W[w] for w,v in topN if w in W ]
            tmp = np.array(vecs)
            min_vec = tmp.min(axis=0)
            max_vec = tmp.max(axis=0)
            avg_vec = tmp.sum(axis=0) / float(len(vecs))

            doc_vec = np.concatenate([min_vec,max_vec,avg_vec])

            features.append( (dt,doc_vec) )
    #assert len(features) > 0
    return features



W = load_word2vec('../../resources/word2vec/mimic10.vec')



if __name__ == '__main__':
    main()
