
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


'''
code_dir = dirname(dirname(os.path.abspath(__file__)))
umlscode_dir = os.path.join(code_dir, 'umls')
if umlscode_dir not in sys.path:
    sys.path.append(umlscode_dir)
import umls_lookup
'''




def get_data(datatype, mode):
    filename = '../../data/structured_%s_%s.pickle' % (mode,datatype)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        outcomes = pickle.load(f)

    assert sorted(X.keys()) == sorted(outcomes.keys())
    return X, outcomes



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

    '''
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
    '''

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

    V = set(range(P.shape[1]))
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





def make_bow(toks):
    # collect all words
    bow = defaultdict(int)
    for w in toks:
        bow[w] += 1
    # only return words that have CUIs
    cui_bow = bow
    '''
    ##cui_bow = defaultdict(int)
    cui_bow = {}
    for w,c in bow.items():
        """
        if umls_lookup.cui_lookup(w):
            cui_bow[w] = 1
        """
        #"""
        for cui in umls_lookup.cui_lookup(w):
            cui_bow[cui] = 1
        #"""
    '''
    return cui_bow



def tokenize(text):
    text = text.lower()
    text = re.sub('[\',\.\-/\n]', ' ', text)
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = re.sub('\d[\d ]+', ' 0 ', text)
    return text.split()



def normalize_y(label, task):
    if task == 'ethnicity':
        if 'WHITE' in label: return 'WHITE'
        return 'NONWHITE'
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
    elif task == 'diagnosis':
        if label is None: return None
        if 'CORONARY ARTERY DISEASE' in label: return 'CORONARY ARTERY DISEASE'
        return label
    elif task == 'admission_location':
        if label == 'CLINIC REFERRAL/PREMATURE': return 'CLINIC REFERRAL/PREMATURE'
        if label == 'EMERGENCY ROOM ADMIT': return 'EMERGENCY ROOM ADMIT'
        if label == 'PHYS REFERRAL/NORMAL DELI': return 'PHYS REFERRAL/NORMAL DELI'
        if label == 'TRANSFER FROM HOSP/EXTRAM': return 'TRANSFER FROM HOSP/EXTRAM'
        return '**ignore**'
    elif task == 'discharge_location':
        if 'HOME' in label: return 'HOME'
        if label == 'SNF': return 'SNF'
        if label == 'REHAB/DISTINCT PART HOSP': return 'REHAB/DISTINCT PART HOSP'
        if label == 'DEAD/EXPIRED': return 'DEAD/EXPIRED'
        return '**ignore**'
    return label



def build_df(notes, hours):
    df = {}
    for pid,records in notes.items():
        pid_bow = {}
        for note in records:
            dt = note[0]
            #print dt
            if isinstance(dt, pd._libs.tslib.NaTType): continue
            if note[0] < datetime.timedelta(days=hours/24.0):
                # access the note's info
                section = note[1]
                toks = tokenize(note[2])

                bow = make_bow(toks)
                for w in bow.keys():
                    pid_bow[w] = 1
        for w in pid_bow.keys():
            df[w] = 1
    return df



def extract_text_features(notes, hours, N, df):
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

            # select top-20 words by tfidf
            tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
            topN = sorted(tfidf.items(), key=lambda t:t[1])[-N:]

            #'''
            # unigram features
            #for w,tf in bow.items():
            for w,tf in topN:
                featname = w
                """
                if section:
                    featname = (w, section) # ('unigram', w, section)
                else:
                    featname = w # ('unigram', w)
                """
                #features[featname] += tf
                #features[featname] += 1
                features[featname] = 1
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


def filter_task(Y, task, per_task_criteria=None):

    # If it's a diagnosis, then only include diagnoses that occur >= 10 times
    if task == 'diagnosis':
        if per_task_criteria is None:
            # count diagnosis frequency
            counts = defaultdict(int)
            for y in Y.values():
                normed = normalize_y(y[task], task)
                if normed == '**ignore**': continue
                counts[normed] += 1

            '''
            for y,c in sorted(counts.items(), key=lambda t:t[1]):
                print '%4d %s' % (c,y)
            exit()
            '''

            # only include diagnois that are frequent enough
            diagnoses = {}
            for y,count in sorted(counts.items(), key=lambda t:t[1])[-5:]:
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
            if task == 'age':
                scores = [ (y if y<90 else 90) for y in scores ]

                mu = np.mean(scores)
                std = np.std(scores)
                thresholds = []
                thresholds.append(mu-1.0*std)
                thresholds.append(mu+1.0*std)
                thresholds.append(float('inf'))
            elif task == 'los':
                '''
                scores = [ (y if y<90 else 90) for y in scores ]

                mu = np.mean(scores)
                std = np.std(scores)
                '''
                thresholds = []
                thresholds.append(1.5)
                thresholds.append(3.5)
                thresholds.append(float('inf'))
            elif task == 'sapsii':
                scores = [ (y if y<90 else 90) for y in scores ]

                mu = np.mean(scores)
                std = np.std(scores)
                thresholds = []
                thresholds.append(mu-0.8*std)
                thresholds.append(mu+0.8*std)
                thresholds.append(float('inf'))

            '''
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

            '''
            print thresholds
            bins = defaultdict(list)
            for s in scores:
                for i in range(len(thresholds)):
                    if s < thresholds[i]:
                        bins[i].append(s)
                        break
            for bin_id,sbin in bins.items():
                print bin_id, len(sbin)
            exit()
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
    Y[-1] = len(per_task_criteria)
    return Y, per_task_criteria





if __name__ == '__main__':
    main()
