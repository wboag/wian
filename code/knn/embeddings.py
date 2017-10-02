
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


codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
structured = os.path.join(codedir, 'structured')
if structured not in sys.path:
    sys.path.append(structured)

from tools import make_bow
from tools import get_data, build_df, extract_text_features, filter_task, load_word2vec



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '2400'], hours_s
    hours = float(hours_s)

    N = int(sys.argv[3])

    train_notes, train_outcomes = get_data('train', mode)
    dev_notes  ,   dev_outcomes = get_data('dev'  , mode)

    # vectorize notes
    train_text_features, df  = extract_features_from_notes(train_notes, hours, N,df=None)
    dev_text_features  , df_ = extract_features_from_notes(  dev_notes, hours, N,df=df)
    assert df == df_

    # throw out any data points that do not have any notes
    good_train_ids = [pid for pid,feats in train_text_features.items() if feats]
    good_dev_ids   = [pid for pid,feats in   dev_text_features.items() if feats]

    def filter_ids(items, ids):
        return {k:v for k,v in items.items() if k in ids}

    train_text_features = filter_ids(train_text_features, good_train_ids)
    train_outcomes      = filter_ids(train_outcomes     , good_train_ids)

    dev_text_features   = filter_ids(  dev_text_features,  good_dev_ids)
    dev_outcomes      = filter_ids(dev_outcomes     , good_dev_ids)

    # vecotrize notes
    train_ids = train_text_features.keys()
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



def extract_text_features(notes, hours, N, df):
    features = []
    for note in notes[:24]:
        dt = note[0]
        #print dt
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if note[0] < datetime.timedelta(days=hours/24.0):
            # access the note's info
            section = note[1]
            toks = note[2]

            bow = make_bow(toks)

            # select top-20 words by tfidf
            tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
            tfidf_in = {k:v for k,v in tfidf.items() if k in W}
            topN = sorted(tfidf_in.items(), key=lambda t:t[1])[-N:]

            if len(topN) < 1:
                continue

            vecs = [ W[w] for w,v in topN if w in W ]
            tmp = np.array(vecs)
            min_vec = tmp.min(axis=0)
            max_vec = tmp.max(axis=0)
            avg_vec = tmp.sum(axis=0) / float(len(vecs))

            doc_vec = np.concatenate([min_vec,max_vec,avg_vec])

            features.append( (dt,doc_vec) )
    #assert len(features) > 0
    return features




def extract_features_from_notes(notes, hours, N, df=None):
    features_list = {}

    # no-op
    if df is None:
        df = build_df(notes, hours)

    # compute features
    for pid,records in sorted(notes.items()):
        features = extract_text_features(records, hours, N, df)
        features_list[pid] = features

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    dimensions = features_list.values()[0][0][1].shape[0]
    features_list[-1] = [(datetime.timedelta(days=0),np.zeros(dimensions))]

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

    #assert sorted(ids) == sorted(text_features.keys())
    dimensions = text_features.values()[0][0][1].shape[0]
    doc_embeddings = defaultdict(list)
    for i,pid in enumerate(ids):
        assert len(text_features[pid])>0, pid
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            doc_embeddings[pid].append(centroid)
    doc_embeddings = dict(doc_embeddings)
    #assert sorted(ids) == sorted(doc_embeddings.keys())

    # agrregate document centroids
    dimensions = text_features.values()[0][0][1].shape[0]
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

    #exit()

    return X, vectorizers


W = load_word2vec('../../resources/word2vec/mimic10.vec')



if __name__ == '__main__':
    main()
