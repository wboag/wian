
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


from tools import compute_stats_binary, compute_stats_multiclass
from tools import get_data, build_df, extract_text_features, filter_task



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '2400'], hours_s
    hours = float(hours_s)
    hrs = hours

    N = int(sys.argv[3])

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    # vectorize notes
    train_text_features, df  = extract_features_from_notes(train_notes, hours, N,df=None)
    dev_text_features  , df_ = extract_features_from_notes(  dev_notes, hours, N,df=df)
    assert df == df_

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['diagnosis']
    #tasks = ['gender']
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
            results(model,train_ids,train_X_lda,train_Y,hrs,'TRAIN',task,criteria,out_f)
            results(model,  dev_ids,  dev_X_lda,  dev_Y,hrs,'DEV'  ,task,criteria,out_f)

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



def extract_features_from_notes(notes, hours, N, df=None):
    features_list = {}

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    features_list[-1] = {'foo':1}

    # doc freq
    if df is None:
        df = build_df(notes, hours)

    # compute features
    for pid,records in notes.items():
        features = extract_text_features(records, hours, N, df)
        features_list[pid] = features

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



def results(model, ids, X, Y, hours, label, task, labels, out_f):
    criteria, vectorizers, clf, lda = model
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
    methoddir = os.path.join(taskdir, 'lda')
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
