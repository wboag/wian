
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



def get_data(datatype, mode):
    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = '%s/data/pickled/%s_%s.pickle' % (homedir,mode,datatype)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        outcomes = pickle.load(f)

    assert sorted(X.keys()) == sorted(outcomes.keys())
    return X, outcomes



def extract_features_from_notes(notes, topN_tfidf_words, extracter, df=None):
    features_list = {}

    # document frequency for top-N tfidf words
    if df is None:
        df = build_df(notes)

    # compute features
    for pid,records in notes.items():
        features = extract_text_features(records, topN_tfidf_words, df, extracter)
        features_list[pid] = features

    return features_list, df



def make_bow(toks):
    # collect all words
    bow = defaultdict(int)
    for w in toks:
        bow[w] += 1
    return bow



def normalize_y(label, task):
    if task == 'ethnicity':
        if 'WHITE' in label: return 'WHITE'
        if 'MULTI RACE' in label: return '**ignore**'
        if 'UNKNOWN' in label: return '**ignore**'
        if 'UNABLE TO OBTAIN' in label: return '**ignore**'
        if 'PATIENT DECLINED TO ANSWER' in label: return '**ignore**'
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
        if 'INTRACRANIAL HEMORRHAGE' == label: return 'INTRACRANIAL HEMORRHAGE'
        if 'GASTROINTESTINAL BLEED' == label: return 'GASTROINTESTINAL BLEED'
        if 'SEPSIS' == label: return 'SEPSIS'
        if 'PNEUMONIA' == label: return 'PNEUMONIA'
        if 'CORONARY ARTERY DISEASE' == label: return 'CORONARY ARTERY DISEASE'
        return '**ignore**'
    elif task == 'admission_type':
        if label == 'EMERGENCY': return 'URGENT'
        if label == 'URGENT': return 'URGENT'
        if label == 'ELECTIVE': return 'ELECTIVE'
        return '**ignore**'
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



def build_df(notes):
    df = {}
    for pid,records in notes.items():
        pid_bow = {}
        for note in records:
            dt = note[0]
            #print dt
            if isinstance(dt, pd._libs.tslib.NaTType): continue

            # access the note's info
            section = note[1]
            toks = note[2]

            # mark each word's presence in the document
            bow = make_bow(toks)
            for w in bow.keys():
                pid_bow[w] = 1

        # mark each word's presence in the stay
        for w in pid_bow.keys():
            df[w] = 1
    return df



def extract_text_features(notes, topN_tfidf_words, df, extracter):
    # are we returning a feature_dict or a list of vectors?
    if extracter == 'bow':
        features = {}
    elif extracter == 'embeddings':
        features = []
    else:
        raise Exception('unknown extracter: "%s"' % extracter)

    # for each note
    for note in notes:
        dt = note[0]
        assert not isinstance(dt, pd._libs.tslib.NaTType)

        # access the note's info
        section = note[1]
        toks = note[2]

	# extract unigrams from note
        bow = make_bow(toks)

        # select top-20 words by tfidf
        tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
        tfidf_in = {k:v for k,v in tfidf.items() if k in W}
        topN = sorted(tfidf_in.items(), key=lambda t:t[1])[-topN_tfidf_words:]

        if extracter == 'bow':
            # unigram features
            for w,tf in topN:
                featname = w
                features[featname] = 1
        elif extracter == 'embeddings':
            # if there are no words in this note that are top-N tfidf, then it is a zero vector
	    if len(topN) < 1:
                emb_size = W.values()[0].shape[0]
                vecs = [ np.zeros(emb_size) ]
            else:
                vecs = [ W[w] for w,v in topN if w in W ]

	    # aggregate this note into min/max/average word vectors
	    tmp = np.array(vecs)
	    min_vec = tmp.min(axis=0)
	    max_vec = tmp.max(axis=0)
	    avg_vec = tmp.sum(axis=0) / float(len(vecs))

	    doc_vec = np.concatenate([min_vec,max_vec,avg_vec])

	    features.append( (dt,doc_vec) )
        else:
            raise Exception('unknown extracter: "%s"' % extracter)

    return features



def filter_notes_tfidf(notes, N, df):
    filtered_notes = []

    for note in notes[:24]:
        dt = note[0]
        #print dt
        if isinstance(dt, pd._libs.tslib.NaTType): continue

        # access the note's info
        section = note[1]
        toks = note[2]

        bow = make_bow(toks)

        # select top-20 words by tfidf
        tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
        tfidf_in = {k:v for k,v in tfidf.items()}
        topN = sorted(tfidf_in.items(), key=lambda t:t[1])[-N:]
        topN = [item[0] for item in topN]

        filtered_note = [w for w in toks if w in topN]
        filtered_notes.append(filtered_note)

    return filtered_notes



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
                #thresholds.append(mu-1.0*std)
                #thresholds.append(mu+1.0*std)
                thresholds.append(50)
                thresholds.append(80)
                thresholds.append(float('inf'))
            elif task == 'los':
                #'''
                scores = [ (y if y<90 else 90) for y in scores ]

                #mu = np.mean(scores)
                #std = np.std(scores)
                #'''
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
                #normed = y[task]
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




def mkdir_p(path):
    prefix = os.path.split(path)[0]
    if not os.path.exists(prefix):
        mkdir_p(prefix)
    if not os.path.exists(path):
        print 'making:', path
        os.mkdir(path)


def load_word2vec(filename):
    W = {}
    with open(filename, 'r') as f:
        for i,line in enumerate(f.readlines()):
            if i==0: continue
            '''
            if sys.argv[1]=='small' and i>=50:
                break
            '''
            toks = line.strip().split()
            w = toks[0]
            vec = np.array(map(float,toks[1:]))
            W[w] = vec
    return W



homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = load_word2vec('%s/resources/mimic10.vec' % homedir)
#W = defaultdict(lambda:np.zeros(300))



if __name__ == '__main__':
    main()
