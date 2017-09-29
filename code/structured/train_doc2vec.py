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
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from tools import compute_stats_multiclass, get_data, compute_stats_binary, filter_task



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    N = sys.argv[3]

    if len(sys.argv)>4 and (sys.argv[4]=='False' or sys.argv[4]=='True'):
	retrain = (sys.argv[4] == 'True')
    else:
        retrain = True
 
    assert hours_s in ['12', '24', '2400'], hours_s
    hours = float(hours_s)

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    print 'Preprocessing notes'
    # put notes in form needed for testing/training doc2vec model
    train_tagged_docs = preprocess_notes(train_notes, hours)
    dev_tagged_docs   = preprocess_notes(  dev_notes, hours)

    doc2vec_name = '../../models/structured/doc2vec/doc2vec_%s.p' % mode
    features_name = '../../models/structured/doc2vec/doc2vec_%s.features' % (mode)
    if retrain or not os.path.isfile(doc2vec_name) or not os.path.isfile(features_name):
        print 'Fitting doc2vec model'
        # Combine notes into a single vector for training doc2vec
        X = [train_tagged_docs[pid] for pid in train_tagged_docs.keys()]
        X = X + [dev_tagged_docs[pid] for pid in dev_tagged_docs.keys()]
        doc2vec = doc2vec_features_fit(X)
        
        print 'Transforming doc2vec model'
        train_doc2vec_feats = doc2vec_features_transform(doc2vec, train_tagged_docs)
        dev_doc2vec_feats = doc2vec_features_transform(doc2vec, dev_tagged_docs)
        F = {'train': train_doc2vec_feats, 'dev': dev_doc2vec_feats}
        
        print 'Saving doc2vec'
        with open(doc2vec_name, 'wb') as f:
            pickle.dump(doc2vec, f)
        with open(features_name, 'wb') as f:
            pickle.dump(F, f)
    else:
        doc2vec = pickle.load(open(doc2vec_name, 'rb'))
        F = pickle.load(open(features_name, 'rb'))
        train_doc2vec_feats = F['train']
        dev_doc2vec_feats = F['dev']
    

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['diagnosis']
    #tasks = ['gender']
    tasks = ['diagnosis', 'los', 'age', 'gender', 'ethnicity', 'admission_type', 'hosp_expire_flag']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit', 'language', 'marital_status', 'insurance'])
    for task in tasks:
        if task in excluded:
            continue
        modelname = '../../models/structured/doc2vec/%s_%s.model' % (mode,task)
        if (not retrain) and os.path.isfile(modelname):
            print 'Loading model for task %s mode %s'%(task,mode)
	    # load serialized model model
            M = pickle.load(open(modelname, 'rb'))
            models[task] = (M['criteria'], M['clf'])
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
        X_doc2vec = [train_doc2vec_feats[pid] for pid in train_ids]

        # learn SVM
        print 'Learning SVM'
        clf = LinearSVC(C=1e1)
        clf.fit(X_doc2vec, Y)

        model = (criteria, clf)
        models[task] = model

    for task,model in models.items():
        criteria, clf = model

        # train data
        train_labels,_ = filter_task(train_outcomes, task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X_doc2vec = [train_doc2vec_feats[pid] for pid in train_ids]
        train_Y = np.array([train_labels[pid] for pid in train_ids])

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        dev_X_doc2vec = [dev_doc2vec_feats[pid] for pid in dev_ids]
        dev_Y = np.array([dev_labels[pid] for pid in dev_ids])

        with io.StringIO() as out_f:
            # analysis
            analyze(clf, criteria, out_f)

            # eval on dev data
            results(model, train_ids, train_X_doc2vec, train_Y, hours, 'TRAIN', task, criteria, out_f)
            results(model,   dev_ids,   dev_X_doc2vec,   dev_Y, hours, 'DEV'  , task, criteria, out_f)

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
            TaggedDocument = gensim.models.doc2vec.TaggedDocument
            preprocessed_notes[pid] = TaggedDocument(concat_note, [pid])

    return preprocessed_notes


def analyze(clf, labels_map, out_f):
    if not isinstance(labels_map, dict):
       return 

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
    criteria, clf = model

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
    criteria, clf = model
    if not isinstance(criteria, dict):
       return 

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


def extract_concat_note(notes, hours):
    concat_note = []     
    for note in notes:
        dt = note[0]
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if dt < datetime.timedelta(days=hours/24.0):
            # access the note's info
            category = note[1]
            tokens = note[2]
            concat_note += tokens
    return concat_note

def extract_notes_list(notes, hours):
    notes_list = []
    for note in notes:
        dt = note[0]
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if dt < datetime.timedelta(days=hours/24.0):
            # access the note's info
            category = note[1]
            tokens = note[2]
            notes_list.append(tokens)
    return notes_list
 
def doc2vec_features_fit(X):
    '''
    docs = []
    for patient in X:
        docs += patient
    '''
    docs = X
    model = Doc2Vec(size=150, min_count=2, iter=100, workers=8)
    print 'building model vocab'
    model.build_vocab(docs)
    print 'training model'
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    return model

def doc2vec_features_transform(model, X):
    '''
    features = []
    for patient in X:
        notes_features = np.array([model.infer_vector(tagged_doc.words) for tagged_doc in patient])
        feat_mean = np.mean(notes_features, axis=0)
        feat_min = np.min(notes_features, axis=0)
        feat_max = np.max(notes_features, axis=0)
        patient_features = np.concatenate([feat_min, feat_max, feat_mean])
        features.append(patient_features)
    '''
    features = {pid:model.infer_vector(X[pid].words) for pid in X.keys()}
    return features
    

def load_file(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text


def dump_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)



if __name__ == '__main__':
    main()
