

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname
from sklearn.decomposition import PCA
from collections import defaultdict
import sys
import math


def main():

    mode = sys.argv[1]
    model = sys.argv[2]
    N = int(sys.argv[3])

    # load the data
    filename = '../../models/knn/knn_%s_%s-%d.pickle' % (mode,model,N)
    print 'loading:', filename
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    ids = model['ids']
    X = model['X']
    X2 = model['X2']

    outcomes = model['outcomes']

    try:
        query_id = int(sys.argv[3])
        ind = ids.index(query_id)
    except Exception, e:
        print 'bad query. please select from ids'
        print ids
        exit(1)

    # get the query document
    query = X[ind,:]
    query /= dot(query, query)**0.5

    print 'QUERY:', query_id
    pretty_print(query_id)
    print '\n\n\n\n'

    # how close is each document?
    scores = [ cosine(query, X[i,:]) for i in range(X.shape[0]) ]

    # look at closest documents
    N = 3
    for pid,score in sorted(zip(ids,scores), key=lambda t:t[1])[-N-1:-1]:
        print 'PID:', pid
        print 'SCORE:', score
        pretty_print(pid)
        print '\n\n\n\n'



def pretty_print(pid, N=120):
    home_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
    notes_dir = os.path.join(home_dir, 'data', 'readable')
    filename = os.path.join(notes_dir, '%d.txt' % pid)
    with open(filename, 'r') as f:
        text = f.read()

    preamble = text.split('===================================')[0]
    indent_print(preamble, N)



def indent_print(text, N):
    lines = text.split('\n')
    for line in lines:
        if len(line) < N:
            print '\t\t', line
        else:
            for i in range(int(math.ceil( float(len(line))/N  ))):
                firstN = line[:N]
                print '\t\t', firstN
                line = line[N:]
            


def cosine(unit, v):
    return dot(unit,v) / (dot(v,v) + 1e-9)**0.5

def dot(u,v):
    if type(u) == type(np.array([])):
        ans = np.dot(u,v)
    else:
        ans = (u * v.T)[0,0]
    return ans



if __name__ == '__main__':
    main()
