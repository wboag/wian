
import numpy as np
import sys
import os
from os.path import dirname
import pandas as pd
import math
import datetime
import re
import cPickle as pickle
from collections import defaultdict

from sklearn.metrics import roc_auc_score



def results_svm(model, ids, X, Y, label, task, labels, out_f):
    criteria, _, clf = model

    # for AUC
    P_ = clf.decision_function(X)

    # sklearn has stupid changes in API when doing binary classification. make it conform to 3+
    if len(labels)==2:
        m = X.shape[0]
        P = np.zeros((m,2))
        P[:,0] = -P_
        P[:,1] =  P_
    else:
        P = P_

    # hard predictions
    train_pred = P.argmax(axis=1)

    # what is the predicted vocab without the dummy label?
    if task in ['los','age','sapsii']:
        V = range(len(labels))
    else:
        V = labels.keys()

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if len(V) == 2:
        scores = P[:,1] - P[:,0]
        compute_stats_binary(task, train_pred, scores, Y, criteria, out_f)
    else:
        compute_stats_multiclass(task,train_pred,P,Y,criteria,out_f)
    out_f.write(unicode('\n\n'))



def results_onehot_keras(model, ids, X, onehot_Y, label, task, out_f):
    criteria, num_docs, lstm_model = model

    # for AUC
    P = lstm_model.predict(X)

    # hard predictions
    train_pred = P.argmax(axis=1)

    Y = onehot_Y.argmax(axis=1)
    num_tags = P.shape[1]

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if num_tags == 2:
        scores = P[:,1] - P[:,0]
        compute_stats_binary(task, train_pred, scores, Y, criteria, out_f)
    else:
        compute_stats_multiclass(task, train_pred, P, Y, criteria, out_f)
    out_f.write(unicode('\n\n'))



def compute_stats_binary(task, pred, P, ref, labels, out_f):
    # santiy check
    assert all(map(int,P>0) == pred)

    V = [0,1]
    n = len(V)
    assert n==2, 'sorry, must be exactly two labels (how else would we do AUC?)'
    conf = np.zeros((n,n), dtype='int32')
    for p,r in zip(pred,ref):
        conf[p][r] += 1

    out_f.write(unicode(conf))
    out_f.write(unicode('\n'))

    tp = conf[1,1]
    tn = conf[0,0]
    fp = conf[1,0]
    fn = conf[0,1]

    precision   = tp / (tp + fp + 1e-9)
    recall      = tp / (tp + fn + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    f1 = (2*precision*recall) / (precision+recall+1e-9)

    tpr =  true_positive_rate(pred, ref)
    fpr = false_positive_rate(pred, ref)

    accuracy = (tp+tn) / (tp+tn+fp+fn + 1e-9)

    out_f.write(unicode('\tspecificity %.3f\n' % specificity))
    out_f.write(unicode('\tsensitivty: %.3f\n' % sensitivity))

    auc = roc_auc_score(ref, P)
    out_f.write(unicode('\t\t\tauc: %.3f\n' % auc))

    out_f.write(unicode('\taccuracy:   %.3f\n' % accuracy   ))
    out_f.write(unicode('\tprecision:  %.3f\n' % precision  ))
    out_f.write(unicode('\trecall:     %.3f\n' % recall     ))
    out_f.write(unicode('\tf1:         %.3f\n' % f1         ))
    out_f.write(unicode('\tTPR:        %.3f\n' % tpr        ))
    out_f.write(unicode('\tFPR:        %.3f\n' % fpr        ))



def compute_stats_multiclass(task, pred, P, ref, labels_map, out_f):
    # santiy check
    assert all(map(int,P.argmax(axis=1)) == pred)

    # confusion matrix
    V = set(range(P.shape[1]))
    n = len(set((V)))
    conf = np.zeros((n,n), dtype='int32')
    for p,r in zip(pred,ref):
        conf[p][r] += 1

    # task labels (for printing results)
    if task in ['sapsii', 'age', 'los']:
        labels = []
        labels_ = [0] + labels_map
        for i in range(len(labels_)-1):
            label = '[%s,%s)' % (labels_[i],labels_[i+1])
            labels.append(label)
    else:
        labels = [label for label,i in sorted(labels_map.items(), key=lambda t:t[1])]

    out_f.write(unicode(conf))
    out_f.write(unicode('\n'))

    # compute P, R, F1
    precisions = []
    recalls = []
    f1s = []
    out_f.write(unicode('\t prec  rec    f1   label\n'))
    for i in range(n):
        label = labels[i]

        tp = conf[i,i]
        pred_pos = conf[i,:].sum()
        ref_pos  = conf[:,i].sum()

        precision   = tp / (pred_pos + 1e-9)
        recall      = tp / (ref_pos + 1e-9)
        f1 = (2*precision*recall) / (precision+recall+1e-9)

        out_f.write(unicode('\t%.3f %.3f %.3f %s\n' % (precision,recall,f1,label)))

        # Save info
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall    = sum(recalls   ) / len(recalls   )
    avg_f1        = sum(f1s       ) / len(f1s       )
    out_f.write(unicode('\t--------------------------\n'))
    out_f.write(unicode('\t%.3f %.3f %.3f avg\n' % (avg_precision,avg_recall,avg_f1)))



def true_positive_rate(pred, ref):
    tp,fn = 0,0
    for p,r in zip(pred,ref):
        if p==1 and r==1:
            tp += 1
        elif p==0 and r==1:
            fn += 1
    return tp / (tp + fn + 1e-9)


def false_positive_rate(pred, ref):
    fp,tn = 0,0
    for p,r in zip(pred,ref):
        if p==1 and r==0:
            fp += 1
        elif p==0 and r==0:
            tn += 1
    return fp / (fp + tn + 1e-9)

