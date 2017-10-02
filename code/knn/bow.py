
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

from tools import get_data, build_df, extract_text_features, filter_task



def main():

    mode = sys.argv[1]

    hours_s = sys.argv[2]
    assert hours_s in ['12', '24', '240', '2400'], hours_s
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



if __name__ == '__main__':
    main()
