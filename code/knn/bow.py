
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
from sklearn.decomposition import PCA




def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '240'], hours_s
    hours = float(hours_s)

    N = int(sys.argv[3])

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    # vectorize notes
    train_text_features, df  = extract_features_from_notes(train_notes, hours, N,df=None)
    dev_text_features  , df_ = extract_features_from_notes(  dev_notes, hours, N,df=df)
    assert df == df_

    # vecotrize notes
    train_ids = train_notes.keys()
    X, vectorizers = vectorize_X(train_ids, train_text_features, vectorizers=None)

    # PCA to decompose into base components
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X.todense())

    # for now, pickle it all so I can play with it locally
    filename = '../../models/knn/knn_%s_bow-%d.pickle' % (mode,N)
    print 'serializing:', filename
    model = {'ids':train_ids, 'X':X, 'X2':X2, 'outcomes':train_outcomes}
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    '''
    # extract labels into list
    Y = np.array([train_Y[pid] for pid in train_ids])
    assert sorted(set(Y)) == range(max(Y)+1), 'need one of each example'
    '''


def get_data(datatype, mode):
    filename = '../../data/structured_%s_%s.pickle' % (mode,datatype)
    with open(filename, 'rb') as f:
        X = pickle.load(f)
        outcomes = pickle.load(f)

    assert sorted(X.keys()) == sorted(outcomes.keys())
    return X, outcomes




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



def make_bow(toks):
    bow = defaultdict(int)
    for w in toks:
        bow[w] += 1
    return bow



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
            topN = sorted(tfidf.items(), key=lambda t:t[1])[:N]

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



if __name__ == '__main__':
    main()
