
import numpy as np
import sys
import os
from os.path import dirname


code_dir = dirname(dirname(os.path.abspath(__file__)))
umlscode_dir = os.path.join(code_dir, 'umls')
if umlscode_dir not in sys.path:
    sys.path.append(umlscode_dir)
import umls_lookup



def compute_stats_binary(task, pred, P, ref, labels, out_f):
    # santiy check
    assert all(map(int,P>0) == pred)

    V = set(pred) | set(ref)
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

    '''
    print conf
    print 'p: ', precision
    print 'r: ', recall
    print 'sn:', sensitivity
    print 'sp: ', specificity
    print 'tpr:   ', true_positive_rate(pred, ref)
    print '1-fpr: ', 1 - false_positive_rate(pred, ref)
    exit()
    '''

    out_f.write(unicode('\tspecificity %.3f\n' % specificity))
    out_f.write(unicode('\tsensitivty: %.3f\n' % sensitivity))

    success = 0.
    pos_examples = [p for p,y in zip(P, ref) if y==1]
    neg_examples = [p for p,y in zip(P, ref) if y==0]
    trials  = len(pos_examples) * len(neg_examples) + 1e-9
    for pos in pos_examples:
        for neg in neg_examples:
            if pos > neg:
                success += 1
    empirical_auc = success / trials
    out_f.write(unicode('\t\temp auc:    %.3f\n' % empirical_auc))

    # AUC
    tprs,fprs = [], []
    smidge = min(P)*.01
    for threshold in np.linspace(min(P)-smidge,max(P)+smidge,1001):
        vals = np.array(map(int,P>threshold))
        tpr_ =  true_positive_rate(vals, ref)
        fpr_ = false_positive_rate(vals, ref)

        tprs.append(tpr_)
        fprs.append(fpr_)

    auc = 0.0
    for i in range(len(tprs)-1):
        assert fprs[i] >= fprs[i+1]
        avg_y = .5 * (tprs[i] + tprs[i+1])
        dx = fprs[i] - fprs[i+1]
        #auc += dx * dy
        auc += avg_y * dx
    out_f.write(unicode('\t\tauc:        %.3f\n' % auc))

    out_f.write(unicode('\taccuracy:   %.3f\n' % accuracy   ))
    out_f.write(unicode('\tprecision:  %.3f\n' % precision  ))
    out_f.write(unicode('\trecall:     %.3f\n' % recall     ))
    out_f.write(unicode('\tf1:         %.3f\n' % f1         ))
    out_f.write(unicode('\tTPR:        %.3f\n' % tpr        ))
    out_f.write(unicode('\tFPR:        %.3f\n' % fpr        ))

    '''
    print fprs
    print tprs
    '''



def compute_stats_multiclass(task, pred, P, ref, labels_map, out_f):
    # santiy check
    assert all(map(int,P.argmax(axis=1)) == pred)

    # get rid of that final prediction dimension
    #pred = pred[1:]
    #ref  =  ref[1:]

    V = set(pred) | set(ref)
    n = max(V)+1
    conf = np.zeros((n,n), dtype='int32')
    for p,r in zip(pred,ref):
        conf[p][r] += 1


    if task in ['sapsii', 'age', 'los']:
        labels = []
        labels_ = [0] + labels_map
        for i in range(len(labels_)-1):
            label = '[%d,%s)' % (labels_[i],labels_[i+1])
            labels.append(label)
    else:
        labels = [label for label,i in sorted(labels_map.items(), key=lambda t:t[1])]


    out_f.write(unicode(conf))
    out_f.write(unicode('\n'))

    precisions = []
    recalls = []
    f1s = []
    out_f.write(unicode('\t prec  rec    f1   label\n'))
    for i in range(n-1):
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



if __name__ == '__main__':
    main()
