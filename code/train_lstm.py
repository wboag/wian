
import os, sys
import re
import math
import cPickle as pickle
import io
import numpy as np
import random
import time

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
from keras.layers.core import Masking
from keras.callbacks import Callback

from compute_results import results_onehot_keras
from tools import get_data, normalize_y, filter_task, mkdir_p
from tools import extract_features_from_notes
from tools import W



def main():

    try:
        size = sys.argv[1]
    except Exception as e:
        print e
        print '\n\tusage: python %s <small|all>\n' % sys.argv[0]
        exit()

    # hyperparameter
    topN_tfidf_words = 20

    train_notes, train_outcomes = get_data('train', size)
    test_notes ,  test_outcomes = get_data('test' , size)

    # max number of documents (this is used during vectorization)
    num_docs = max(map(len,train_notes.values()))

    # extract feature lists
    train_text_features, df  = extract_features_from_notes(train_notes, topN_tfidf_words, 'embeddings', df=None)
    test_text_features , df_ = extract_features_from_notes( test_notes, topN_tfidf_words, 'embeddings', df=df)
    assert df == df_

    # Fit model for each prediction task
    tasks = ['ethnicity', 'age', 'admission_type', 'hosp_expire_flag', 'gender', 'los', 'diagnosis']
    #tasks = ['diagnosis']
    for task in tasks:

        print 'task:', task

        ### Train model

        # extract appropriate data
        train_Y, criteria = filter_task(train_outcomes, task, per_task_criteria=None)
        train_ids = sorted(train_Y.keys())
        print 'train examples:', len(train_Y)

        # vecotrize notes
        train_X = vectorize_X(train_ids, train_text_features, num_docs=num_docs)
        print 'num_features:  ', train_X.shape[1], '\n'

        train_Y = vectorize_Y(train_ids, train_Y, criteria)
        num_tags = train_Y.shape[1]

        # build model
        lstm_model = create_lstm_model(num_docs, num_tags, train_X, train_Y)
        lstm_model.summary()

        # test data
        test_labels,_ = filter_task(test_outcomes, task, per_task_criteria=criteria)
        test_ids = sorted(test_labels.keys())
        test_X = vectorize_X(test_ids, test_text_features, num_docs=num_docs)
        test_Y = vectorize_Y(test_ids, test_labels, criteria)

	# fit model

        # fit model
        filepath="/tmp/weights-%d.best.hdf5" % random.randint(0,10000)
        save_best = SaveBestCallback(filepath)
        lstm_model.fit(train_X, train_Y, epochs=100, verbose=1, batch_size=32, 
                       validation_data=(test_X,test_Y),
                       callbacks=[save_best])
        lstm_model.load_weights(filepath)
        os.remove(filepath)

        model = (criteria, num_docs, lstm_model)

        ### Evaluation

        with io.StringIO() as out_f:
            # analysis
	    pass

            # eval on test data
            results_onehot_keras(model, train_ids, train_X, train_Y, 'TRAIN', task, out_f)
            results_onehot_keras(model,  test_ids,  test_X,  test_Y, 'TEST' , task, out_f)

            output = out_f.getvalue()
        print output

        # error analysis
        error_analysis(model, test_ids, test_notes, test_text_features, test_X, test_Y, 'TEST', task)

        # serialize trained model
        homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        modelname = '%s/models/rnn_%s_%s.model' % (homedir,size,task)
        M = {'criteria':criteria, 'num_docs':num_docs, 'model':lstm_pickle(lstm_model), 'output':output}
        with open(modelname, 'wb') as f:
            pickle.dump(M, f)


def macroaverage_f1(ref, pred):
    num_classes = max(ref.max(), pred.max()) + 1
    conf = np.zeros((num_classes,num_classes))
    for r,p in zip(ref, pred):
        conf[p][r] += 1

    f1s = []
    for i in range(num_classes):
        tp = conf[i,i]
        pred_pos = conf[i,:].sum()
        ref_pos  = conf[:,i].sum()

        precision   = tp / (pred_pos + 1e-9)
        recall      = tp / (ref_pos + 1e-9)
        f1 = (2*precision*recall) / (precision+recall+1e-9)

        f1s.append(f1)

    return sum(f1s) / len(f1s)



def lstm_pickle(lstm):
    # needs to return something pickle-able (so get binary serialized string)
    tmp_file = '/tmp/tmp_keras_weights-%d' % random.randint(0,10000)
    lstm.save_weights(tmp_file)
    with open(tmp_file, 'rb') as f:
        lstm_str = f.read()
    os.remove(tmp_file)
    return lstm_str



def vectorize_Y(ids, y_dict, criteria):
    # extract labels into list
    num_tags = len(criteria)
    Y = np.zeros((len(ids),num_tags))
    for i,pid in enumerate(ids):
        ind = y_dict[pid]
        Y[i,ind] = 1
    return Y



def vectorize_X(ids, text_features, num_docs):
    num_samples = len(ids)
    emb_size = W['and'].shape[0]

    dimensions = text_features.values()[0][0][1].shape[0]
    dts = np.zeros((num_samples,num_docs,1))
    X = np.zeros((num_samples,num_docs,dimensions))
    for i,pid in enumerate(ids):
        for j,(dt,centroid) in enumerate(text_features[pid][:num_docs]):
            # right padding
            dts[i,num_docs-j-1,0] = dt.seconds
            X[i,num_docs-j-1,:] = centroid

    return X



class SaveBestCallback(Callback):
    def __init__(self, filepath):
        self.validation_data = None
        self.model = None
	self.filepath = filepath
        self.best_f1 = float('-inf')

    def on_epoch_end(self, batch, logs={}):
        ref_onehot = self.validation_data[1]
        pred_p     = self.model.predict(self.validation_data[0])

	ref  = ref_onehot.argmax(1)
	pred = pred_p.argmax(1)

	current_f1 = macroaverage_f1(ref, pred)

	if current_f1 > self.best_f1:
            self.best_f1 = current_f1
	    self.model.save_weights(self.filepath)



def create_lstm_model(num_docs, num_tags, X_dts, Y):

    #X,dts = X_dts
    X = X_dts

    emb_size = X.shape[2]

    # document w2v centroids
    X_input  = Input(shape=(num_docs,emb_size) , dtype='float32', name='doc')
    X_masked = Masking(0.0)(X_input)
    seq = Bidirectional(LSTM(256, dropout=0.5))(X_masked) # 512

    # Predict target
    pre_pred = Dense(128, activation='tanh')(seq) # 128
    pre_pred_d = Dropout(0.5)(pre_pred)
    pred     = Dense(num_tags, activation='softmax')(pre_pred_d)

    # Putting it all together
    model = Model(inputs=X_input, outputs=pred)
    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model



def error_analysis(model, ids, notes, text_features, X, onehot_Y, label, task):
    criteria, num_docs, clf = model

    if task in ['sapsii', 'age', 'los']:
        V = {}
        labels_ = [0] + criteria
        for i in range(len(labels_)-1):
            label = '[%d,%s)' % (labels_[i],labels_[i+1])
            V[i] = label
    else:
        V = {v:k for k,v in criteria.items()}

    # for confidence
    #P = clf.predict(list(X))
    P = clf.predict(X)
    pred = P.argmax(axis=1)
    Y = onehot_Y.argmax(axis=1)

    # convert predictions to right vs wrong
    confidence = {}
    for i,scores in enumerate(P.tolist()):
        prediction = pred[i]
        pid = ids[i]
        ind = Y[i]
        confidence[pid] = (scores[ind], scores, prediction, ind)

    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    methoddir = os.path.join(homedir, 'output', task, 'lstm')
    mkdir_p(methoddir)

    # order predictions by confidence
    for pid,conf in sorted(confidence.items(), key=lambda t:t[1]):
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


if __name__ == '__main__':
    main()
