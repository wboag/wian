
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

from FableLite import W




def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    N = int(sys.argv[3])
    assert hours_s in ['12', '24', '240'], hours_s
    hours = float(hours_s)

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
    X2 = pca.fit_transform(X)

    # for now, pickle it all so I can play with it locally
    filename = '../../models/knn/knn_%s_embeddings-%d.pickle' % (mode,N)
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
    emb_size = W['and'].shape[0]

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    features_list[-1] = [(0,np.zeros(emb_size))]

    # no-op
    if df is None:
        df = build_df(notes, hours)

    # compute features
    for pid,records in notes.items():
        features = extract_text_features(records, hours, N, df)
        features_list[pid] = features

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

    doc_embeddings = defaultdict(list)
    for i,pid in enumerate(ids):
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            doc_embeddings[pid].append(centroid)

    # agrregate document centroids
    X = np.zeros((len(ids),emb_size))
    for i,pid in enumerate(ids):
        vecs = doc_embeddings[pid]

        # average
        res = np.zeros((1,emb_size))
        for vec in vecs:
            res += vec
        pid_vector = res / (len(vecs) + 1e-9)

        '''
        # average
        res = np.zeros((1,emb_size))
        for vec in vecs:
            res += vec
        pid_vector = res / (len(vecs) + 1e-9)
        '''

        X[i,:] = pid_vector

    return X, vectorizers



def make_bow(toks):
    bow = defaultdict(int)
    for w in toks:
        if len(w) != 1 or w=='m' or w=='f':
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
    features = []
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

            # compute centroid now (i.e. keras cant fine-tune)
            emb_size = W['and'].shape[0]
            w2v_sum = np.zeros(emb_size)
            N = 0
            for w,v in topN:
                if w in W:
                    #print w
                    w2v_sum +=  W[w]
                    N += 1
            w2v_centroid = w2v_sum / (N+1e-9)
            #print

            features.append( (dt,w2v_centroid) )
    #exit()
    return features



if __name__ == '__main__':
    main()
