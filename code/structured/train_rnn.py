
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


from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.layers.crf import ChainCRF
from keras.layers.wrappers import TimeDistributed
from keras.layers import Masking
from keras.callbacks import EarlyStopping
from keras.layers.pooling import GlobalAveragePooling1D


#from FableLite import W

from tools import compute_stats_multiclass, compute_stats_binary
from tools import get_data, build_df, tokenize, make_bow, normalize_y, filter_task



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

    # Fit model for each prediction task
    models = {}
    tasks = dev_outcomes.values()[0].keys()
    #tasks = ['hosp_expire_flag']
    #tasks = ['diagnosis']
    #tasks = ['gender']
    excluded = set(['subject_id', 'first_wardid', 'last_wardid', 'first_careunit', 'last_careunit', 'language', 'marital_status', 'insurance'])

    for task in tasks:
        if task in excluded:
            continue

        #task = 'admission_location'
        #task = 'diagnosis'

        # extract appropriate data
        train_Y, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_Y.keys())
        print 'task:', task
        print 'N:   ', len(train_Y)

        # vecotrize notes
        X, vectorizers = vectorize_X(train_ids, train_text_features, vectorizers=None)

        Y = vectorize_Y(train_ids, train_Y, criteria)
        num_tags = Y.shape[1]

        # build model
        lstm_model = create_lstm_model(vectorizers, num_tags, X, Y)

        # fit model
        X_doc,X_dts = X
        lstm_model.fit([X_doc,X_dts], Y, epochs=10, verbose=1)

        model = (criteria, vectorizers, lstm_model)
        models[task] = model

    for task,model in models.items():
        criteria, vectorizers, lstm_model = model
        vect = vectorizers[0]

        # train data
        train_labels,_ = filter_task(train_outcomes, task, per_task_criteria=criteria)
        train_ids = sorted(train_labels.keys())
        train_X,_ = vectorize_X(train_ids, train_text_features, vectorizers=vectorizers)
        train_Y = vectorize_Y(train_ids, train_labels, criteria)

        # dev data
        dev_labels,_ = filter_task(dev_outcomes, task, per_task_criteria=criteria)
        dev_ids = sorted(dev_labels.keys())
        dev_X,_ = vectorize_X(dev_ids, dev_text_features, vectorizers=vectorizers)
        dev_Y = vectorize_Y(dev_ids, dev_labels, criteria)

        with io.StringIO() as out_f:
            # analysis

            # eval on dev data
            results(model, train_ids, train_X, train_Y, hours, 'TRAIN', task, out_f)
            results(model,   dev_ids,   dev_X,   dev_Y, hours, 'DEV'  , task, out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, dev_ids, dev_notes, dev_text_features, dev_X, dev_Y, hours, 'DEV', task)

        # serialize trained model
        modelname = '../../models/structured/bow/%s_%s.model' % (mode,task)
        M = {'criteria':criteria, 'vect':vectorizers, 'model':lstm_pickle(lstm_model), 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)



def lstm_pickle(lstm):
    print 'TODO: pickle by saving weights and reading'
    #exit()



def vectorize_Y(ids, y_dict, criteria):
    # extract labels into list
    num_tags = len(criteria)+1
    Y = np.zeros((len(ids),num_tags))
    for i,pid in enumerate(ids):
        ind = y_dict[pid]
        Y[i,ind] = 1
    return Y



def extract_features_from_notes(notes, hours, N, df=None):
    features_list = {}

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    emb_size = W['and'].shape[0]

    # dummy record (prevent SVM from collparsing to single-dimension pred)
    features_list[-1] = [(datetime.timedelta(days=0),np.ones(3*emb_size)*-100)]

    # doc freq
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

    dts = np.zeros((num_samples,num_docs,1))
    X = np.zeros((num_samples,num_docs,3*emb_size))
    for i,pid in enumerate(ids):
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            # right padding
            dts[i,num_docs-j-1,0] = dt.seconds
            X[i,num_docs-j-1,:] = centroid

    print 'X:', X.shape

    return (X,dts), vectorizers



def create_lstm_model(vectorizers, num_tags, X_dts, Y):

    X,dts = X_dts

    num_docs = vectorizers[0]
    emb_size = X.shape[2]

    print X.shape
    print Y.shape

    # document w2v centroids
    X_input  = Input(shape=(num_docs,emb_size) , dtype='float32', name='doc')
    seq_in = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(X_input)
    seq    = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(seq_in)

    # delta t of timestamps
    dt_input = Input(shape=(num_docs,1), dtype='float32', name='dt')

    # combine inputs
    combined = Concatenate()([seq, dt_input])
    combined_seq = Bidirectional(LSTM(128, dropout=0.5))(combined)

    # Predict target
    pred_in = Dense(64, activation='relu')(combined_seq)
    pred    = Dense(num_tags, activation='softmax')(pred_in)

    # Putting it all together
    model = Model(inputs=[X_input,dt_input], outputs=pred)
    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model



def results(model, ids, X, onehot_Y, hours, label, task, out_f):
    criteria, vectorizers, lstm_model = model
    vect = vectorizers[0]

    # for AUC
    P = lstm_model.predict(list(X))[:,:-1]

    train_pred = P.argmax(axis=1)
    Y = onehot_Y.argmax(axis=1)
    num_tags = P.shape[1]

    out_f.write('%s %s' % (unicode(label),task))
    out_f.write(unicode('\n'))
    if num_tags == 2:
        scores = P[1:,1] - P[1:,0]
        compute_stats_binary(task, train_pred[1:], scores, Y[1:], criteria, out_f)
    else:
        compute_stats_multiclass(task, train_pred[1:], P[1:,:], Y[1:], criteria, out_f)
    out_f.write(unicode('\n\n'))



def error_analysis(model, ids, notes, text_features, X, onehot_Y, hours, label, task):
    criteria, vectorizers, clf = model
    vect = vectorizers[0]

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
    P = clf.predict(list(X))
    pred = P.argmax(axis=1)
    Y = onehot_Y.argmax(axis=1)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    thisdir = os.path.dirname(os.path.abspath(__file__))
    taskdir = os.path.join(thisdir, 'output', task)
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)
    methoddir = os.path.join(taskdir, 'bow')
    if os.path.exists(methoddir):
        shutil.rmtree(methoddir)
    os.mkdir(methoddir)

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
        if pid == -1: continue
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



def softmax(scores):
    e = np.exp(scores)
    return e / e.sum()



def extract_text_features(notes, hours, N, df):
    features = []
    for note in notes:
        dt = note[0]
        if isinstance(dt, pd._libs.tslib.NaTType): continue
        if note[0] < datetime.timedelta(days=hours/24.0):
            # access the note's info
            section = note[1]
            toks = tokenize(note[2])

            bow = make_bow(toks)

            if len(bow) < 10:
                continue

            # select top-20 words by tfidf
            tfidf = { w:tf/(math.log(df[w])+1) for w,tf in bow.items() if (w in df)}
            tfidf_in = {k:v for k,v in tfidf.items() if k in W}
            topN = sorted(tfidf_in.items(), key=lambda t:t[1])[-N:]

            vecs = [ W[w] for w,v in topN if w in W ]
            tmp = np.array(vecs)
            min_vec = tmp.min(axis=0)
            max_vec = tmp.max(axis=0)
            avg_vec = tmp.sum(axis=0) / float(len(vecs))

            doc_vec = np.concatenate([min_vec,max_vec,avg_vec])

            '''
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
            '''

            features.append( (dt,doc_vec) )
    #exit()
    return features



def load_word2vec(filename):
    W = {}
    with open(filename, 'r') as f:
        for i,line in enumerate(f.readlines()):
            '''
            if sys.argv[1]=='small' and i>=50:
                break
            '''
            toks = line.strip().split()
            w = toks[0]
            vec = np.array(map(float,toks[1:]))
            W[w] = vec
    return W



W = load_word2vec('../../resources/word2vec/mimic10.vec')



if __name__ == '__main__':
    main()
