
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
import datetime


from compute_results import results_svm
from tools import get_data, build_df, extract_features_from_notes, filter_task
from tools import mkdir_p



def main():

    try:
        size = sys.argv[1]
    except Exception as e:
        print e
        print '\n\tusage: python %s <small|all>\n' % sys.argv[0]
        exit()

    # hyperparameter
    topN_tfidf_words = 20

    # get data
    train_notes, train_outcomes = get_data('train', size)
    test_notes ,  test_outcomes = get_data('test' , size)

    # vectorize notes (only need to do this once, not each task)
    train_text_features, df  = extract_features_from_notes(train_notes, topN_tfidf_words, 'bow', df=None)
    test_text_features , df_ = extract_features_from_notes( test_notes, topN_tfidf_words, 'bow', df=df)
    assert df == df_

    '''
    for pid,notes in train_notes.items():
        train_text_features[pid]['<num_notes>'] = len(notes)
    for pid,notes in test_notes.items():
        test_text_features[pid]['<num_notes>'] = len(notes)
    '''

    # Make a model for each task
    tasks = ['ethnicity', 'age', 'admission_type', 'hosp_expire_flag', 'gender', 'los', 'diagnosis']
    for task in tasks:

        print 'task:', task

        ### Train model

        # extract appropriate data
        train_labels, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_labels.keys())
        print 'train examples:', len(train_labels)

        # vecotrize notes
        train_X, vect = vectorize_X(train_ids, train_text_features, vect=None)
        print 'num_features:  ', train_X.shape[1], '\n'

        # extract labels into list
        train_Y = np.array([train_labels[pid] for pid in train_ids])
        assert sorted(set(train_Y)) == range(max(train_Y)+1), 'need one of each example'

        # learn SVM
        clf = LinearSVC(C=1e-2)
        clf.fit(train_X, train_Y)

        model = criteria, vect, clf

        ### Evaluate model

        # test data
        test_labels,_ = filter_task(test_outcomes, task, per_task_criteria=criteria)
        test_ids = sorted(test_labels.keys())
        test_X,_ = vectorize_X(test_ids, test_text_features, vect=vect)
        test_Y = np.array([test_labels[pid] for pid in test_ids])

        with io.StringIO() as out_f:
            # analysis
            analyze(task, vect, clf, criteria, out_f)

            # eval on test data
            results_svm(model, train_ids, train_X, train_Y, 'TRAIN', task, criteria, out_f)
            results_svm(model,  test_ids,  test_X,  test_Y, 'TEST' , task, criteria, out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, test_ids, test_notes, test_text_features, test_X, test_Y, 'TEST', task)

        # serialize trained model
        homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        modelname = '%s/models/bow_%s_%s.model' % (homedir,size,task)
        M = {'criteria':criteria, 'vect':vect, 'clf':clf, 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def vectorize_X(ids, text_features, vect):
    # learn vectorizer on training data
    if vect is None:
        vect = DictVectorizer()
        vect.fit(text_features.values())

    # Get the patients we can predict for this task
    filtered_features = [ text_features[pid] for pid in ids ]

    # vectorize the features list into a design matrix
    X = vect.transform(filtered_features)
    return X, vect



def analyze(task, vect, clf, labels_map, out_f):

    ind2feat =  { i:f for f,i in vect.vocabulary_.items() }

    # create a 2-by-m matrix for biary, because sklearn stupidly changes API for binary
    if len(labels_map) == 2:
        n = clf.coef_.shape[1]
        coef_ = np.zeros((2,n))
        coef_[0,:] = -clf.coef_[0,:]
        coef_[1,:] =  clf.coef_[0,:]
    else:
        coef_ = clf.coef_


    # Get the labels that were predicted
    if task in ['sapsii', 'age', 'los']:
        labels = []
        labels_ = [0] + labels_map
        for i in range(len(labels_)-1):
            label = '[%d,%s)' % (labels_[i],labels_[i+1])
            labels.append(label)
    else:
        labels = [label for label,i in sorted(labels_map.items(), key=lambda t:t[1])]

    # most informative features
    out_f = sys.stdout
    out_f.write(unicode(coef_.shape))
    out_f.write(unicode('\n\n'))
    num_feats = 15
    informative_feats = np.argsort(coef_)

    for i,label in enumerate(labels):

        neg_features = informative_feats[i,:num_feats ]
        pos_features = informative_feats[i,-num_feats:]

        # display what each feature is
        out_f.write(unicode('POS %s\n' % label))
        for feat in reversed(pos_features):
            val = coef_[i,feat]
            word = ind2feat[feat]
            if val > 1e-4:
                out_f.write(unicode('\t%-25s: %7.4f\n' % (word,val)))
            else:
                break
        out_f.write(unicode('NEG %s\n' % label))
        for feat in reversed(neg_features):
            val = coef_[i,feat]
            word = ind2feat[feat]
            if -val > 1e-4:
                out_f.write(unicode('\t%-25s: %7.4f\n' % (word,val)))
            else:
                break
        out_f.write(unicode('\n\n'))




def error_analysis(model, ids, notes, text_features, X, Y, label, task):
    criteria, vect, clf = model

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

        n = clf.coef_.shape[1]
        coef_ = np.zeros((2,n))
        coef_[0,:] = -clf.coef_[0,:]
        coef_[1,:] =  clf.coef_[0,:]
    else:
        P = P_
        coef_ = clf.coef_

    # hard predictions
    pred = clf.predict(X)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    methoddir = os.path.join(homedir, 'output', task, 'bow')
    mkdir_p(methoddir)

    def importance(featname, p):
        if featname in vect.vocabulary_:
            ind = vect.vocabulary_[featname]
            return coef_[p,ind]
        else:
            return float('-inf')

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
        if conf[2] == conf[3]:
            success = ''
        else:
            success = '_'
        #if success == '_':
        if True:
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
                for feat,v in sorted(text_features[pid].items(), key=lambda t:importance(t[0],conf[3])):
                #for feat,v in sorted(text_features[pid].items(), key=lambda t:t[1]):
                    imp = importance(feat, conf[2])
                    #imp = v
                    #print >>f, feat, 
                    print >>f, '%.3f %s' % (imp,feat)
                print >>f, '#'*20
                print >>f, 'SCORES'
                pind = conf[2]
                rind = conf[3]
                print >>f, 'predicted:', sum([val*importance(feat,pind) for feat,val in text_features[pid].items() if float('-inf')<importance(feat,pind)<float('inf')])
                print >>f, 'true:     ', sum([val*importance(feat,rind) for feat,val in text_features[pid].items() if float('-inf')<importance(feat,rind)<float('inf')])
                print >>f, '#'*20
                #'''
                for dt,category,text in sorted(notes[pid]):
                    print >>f, dt
                    print >>f, category
                    print >>f, text
                    print >>f, '-'*50
                #'''
                print >>f, ''
                print >>f, '+'*80
                print >>f, ''


if __name__ == '__main__':
    main()
