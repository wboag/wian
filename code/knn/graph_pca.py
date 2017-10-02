

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys


def main():

    mode = sys.argv[1]

    filename = '../../data/knn_%s_bow.pickle' % mode
    print 'loading:', filename
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    ids = model['ids']
    X = model['X']
    X2 = model['X2']
    outcomes = model['outcomes']

    print X.shape, '->', X2.shape
    #exit()

    # extract labels
    filtered_datapoints, view1, view2 = get_labels(ids, outcomes)
    X = X[filtered_datapoints,:]
    X2 = X2[filtered_datapoints,:]

    # get first 2 dim of variation
    d1,d2 = X2.T.tolist()

    fig, [p1,p2] = plt.subplots(1,2)

    p1.scatter(d1, d2, c=np.array(view1))
    p1.set_title('Demographics: Gender and Age')

    p2.scatter(d1, d2, c=np.array(view2))
    p2.set_title('Outcomes: Diagnosis')

    print 'showing'
    plt.show()



def get_labels(ids, outcomes):

    # demographics label
    demographics = {}
    for pid in ids:
        info = outcomes[pid]
        race = info['ethnicity']
        age = info['age']
        gender = info['gender']

        '''
        if race == 'WHITE':
            base_color = [1.0, 0.0, 0.0]      # WHITE    = red
        if race != 'WHITE':
            base_color = [0.0, 0.0, 1.0]      # NONWHITE = blue
        '''

        '''
        if gender == 'M':
            base_color = [1.0, 0.0, 0.0]      # MALE   = red
        if gender != 'M':
            base_color = [0.0, 0.0, 1.0]      # FEMALE = blue
        '''

        #'''
        if race == 'WHITE' and gender == 'M':
            base_color = [1.0, 0.0, 0.0]      # WHITE MALE = red
        if race == 'WHITE' and gender != 'M':
            base_color = [0.0, 1.0, 0.0]      # WHITE LADY = green
        if race != 'WHITE' and gender == 'M':
            base_color = [0.0, 0.0, 0.1]      # NONWHITE MALE = blue
        if race != 'WHITE' and gender != 'M':
            base_color = [1.0, 0.0, 1.0]      # NONWHITE LADY = purple
        #'''

        '''
        MAX = 75.0
        if age >= MAX:
            age = MAX
        intensity = age / MAX
        '''
        intensity = 1

        color = [ c * intensity for c in base_color ]
        demographics[pid] = color

    # outcomes label
    diagnoses = {}
    counts = defaultdict(int)
    for pid in ids:
        info = outcomes[pid]
        diagnosis = info['diagnosis']
        #diagnosis = info['hosp_expire_flag']
        #diagnosis = info['insurance']
        diagnoses[pid] = diagnosis

        counts[diagnosis] += 1

    top5 = sorted(counts.items(), key=lambda t:t[1])[-5:]
    chosen_diagnoses = set([d for d,c in top5])

    colors = ['red', 'green', 'blue', 'magenta', 'yellow']
    encoding = {diagnosis:color for diagnosis,color in zip(chosen_diagnoses,colors)}

    # filter out the ids that dont fit this diagnosis
    filtered_datapoints = []
    filtered_demographics = []
    filtered_diagnoses = []
    for i,pid in enumerate(ids):
        if diagnoses[pid] in chosen_diagnoses:
            filtered_datapoints.append(i)
            filtered_demographics.append(demographics[pid])
            filtered_diagnoses.append(encoding[diagnoses[pid]])

    return (filtered_datapoints, 
            np.array(filtered_demographics), np.array(filtered_diagnoses))



if __name__ == '__main__':
    main()
