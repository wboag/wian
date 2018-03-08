
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

from compute_results import results_svm
from tools import get_data, build_df, make_bow, filter_task, mkdir_p, extract_features_from_notes
from tools import W


def main():

    try:
        size = sys.argv[1]
    except Exception as e:
        print e
        print '\n\tusage: python %s <small|all>\n' % sys.argv[0]
        exit()

    # hyperparameter
    topN_tfidf_words = 20

    train_notes, train_outcomes = get_data('train', size)
    test_notes ,  test_outcomes = get_data('test' , size)

    # max number of documents (this is used during vectorization)
    num_docs = max(map(len,train_notes.values()))

    # extract feature lists
    train_text_features, df  = extract_features_from_notes(train_notes, topN_tfidf_words, 'embeddings', df=None)
    test_text_features , df_ = extract_features_from_notes( test_notes, topN_tfidf_words, 'embeddings', df=df)
    assert df == df_

    # Fit model for each prediction task
    tasks = ['ethnicity', 'age', 'admission_type', 'hosp_expire_flag', 'gender', 'los', 'diagnosis']
    for task in tasks:

        print 'task:', task

        ### Train model

        # extract appropriate data
        train_labels, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_labels.keys())
        print 'train examples:', len(train_labels)

        # vecotrize notes
        train_X = vectorize_X(train_ids, train_text_features, num_docs=num_docs)
        print 'num_features:  ', train_X.shape[1], '\n'

        # extract labels into list
        train_Y = np.array([train_labels[pid] for pid in train_ids])
        assert sorted(set(train_Y)) == range(max(train_Y)+1), 'need one of each example'

        # learn SVM
        clf = LinearSVC(C=1e-2)
        clf.fit(train_X, train_Y)

        model = criteria, num_docs, clf


        ### Evaluate model

        # test data
        test_labels,_ = filter_task(test_outcomes, task, per_task_criteria=criteria)
        test_ids = sorted(test_labels.keys())
        test_X = vectorize_X(test_ids, test_text_features, num_docs=num_docs)
        test_Y = np.array([test_labels[pid] for pid in test_ids])

        with io.StringIO() as out_f:
            # analysis
            #analyze(task, vect, clf, criteria, out_f)

            # eval on test data
            results_svm(model, train_ids, train_X, train_Y, 'TRAIN', task, criteria, out_f)
            results_svm(model,  test_ids,  test_X,  test_Y, 'TEST' , task, criteria, out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, test_ids, test_notes, test_text_features, test_X, test_Y, 'TEST', task)

        # serialize trained model
        modelname = '../models/embeddings_%s_%s.model' % (size,task)
        M = {'criteria':criteria, 'num_docs':num_docs, 'clf':clf, 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def vectorize_X(ids, text_features, num_docs):
    num_samples = len(ids)
    emb_size = W.values()[0].shape[0]

    doc_embeddings = defaultdict(list)
    for i,pid in enumerate(ids):
        assert len(text_features[pid])>0, pid
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            doc_embeddings[pid].append(centroid)
    doc_embeddings = dict(doc_embeddings)

    # agrregate document centroids
    dimensions = text_features.values()[0][0][1].shape[0]
    print 'dim:', dimensions
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

    return X



def error_analysis(model, ids, notes, text_features, X, Y, label, task):
    criteria, num_docs, clf = model

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
    P_ = clf.decision_function(X)

    # sklearn has stupid changes in API when doing binary classification. make it conform to 3+
    if len(criteria)==2:
        m = X.shape[0]
        P = np.zeros((m,2))
        P[:,0] = -P_
        P[:,1] =  P_
    else:
        P = P_

    pred = P.argmax(axis=1)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    methoddir = os.path.join(homedir, 'output', task, 'embeddings')
    mkdir_p(methoddir)

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
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



if __name__ == '__main__':
    main()
