
import os, sys
import cPickle as pickle
from collections import defaultdict
import datetime
import re
import random

import psycopg2
import pandas as pd
import numpy as np

from tools import mkdir_p


# organization: data/$pid.txt (order: demographics, outcome, notes)
homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pickled_dir  = os.path.join(homedir, 'data', 'pickled')
readable_dir = os.path.join(homedir, 'data', 'readable')

# store the pickled and readable notes
mkdir_p(pickled_dir)
mkdir_p(readable_dir)



def main():

    try:
        size = sys.argv[1]
        if size not in ['all','small']:
            raise Exception('bad')
    except Exception, e:
        print '\n\tusage: python %s <all|small>\n' % sys.argv[0]
        exit(1)

    X, Y = gather_data(size)
    assert sorted(X.keys()) == sorted(Y.keys())

    # Randomly shuffle ids to create train/test
    ids = X.keys()
    random.shuffle(ids)

    # make train/test
    train = set()
    dev = set()
    test  = set()

    for i,pid in enumerate(ids):
        n = i % 10
        if n < 7:
            train.add(pid)
        elif n < 9:
            test.add(pid)
        else:
            dev.add(pid)

    print 
    print 'train: %d' % len(train)
    print 'dev:   %d' % len(dev)
    print 'test:  %d' % len(test)
    print 

    categories = defaultdict(int)
    for pid,record in X.items():
        for dt,category,text in record:
            categories[category] += 1

    print
    for cat,count in sorted(categories.items(), key=lambda t:t[1]):
        print '%4d %s' % (count,cat)
    print

    def filter_data(data, ids):
        return {pid:x for pid,x in data.items() if pid in ids}

    train_X = filter_data(X, train)
    dev_X   = filter_data(X, dev)
    test_X  = filter_data(X, test)

    train_Y = filter_data(Y, train)
    dev_Y   = filter_data(Y, dev)
    test_Y  = filter_data(Y, test)

    # how many notes does each patient have
    notes_per_stay = np.array([len(records) for records in X.values()])
    minn  = notes_per_stay.min()
    maxn  = notes_per_stay.max()
    meann = notes_per_stay.mean()
    stdn  = notes_per_stay.std()
    mediann = sorted(notes_per_stay)[len(notes_per_stay)/2]
    print 'notes per day: min=%d max=%d median=%d mean=%.1f std=%.1f\n' % (minn,maxn,mediann,meann,stdn)

    dump_readable(X, Y)

    save_data('%s/%s_train.pickle'% (pickled_dir,size), train_X, train_Y)
    save_data('%s/%s_dev.pickle'  % (pickled_dir,size),   dev_X,   dev_Y)
    save_data('%s/%s_test.pickle' % (pickled_dir,size),  test_X,  test_Y)



def save_data(path, X, Y):
    with open(path, 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(Y, f)
        


def gather_data(size='all'):
    records = {}
    targets = {}

    if size == 'small':
        min_id = 500
        max_id = 1000
    elif size == 'all':
        min_id = 0
        max_id = 1e20
    else:
        raise Exception('bad size "%s"' % size)

    # connect to the mimic database
    con = psycopg2.connect(dbname='mimic')

    # query mimic for ICU stays
    first_icu_query = '''
    select distinct i.subject_id, i.hadm_id,
    i.icustay_id, i.intime, i.outtime, i.admittime, i.dischtime
      FROM mimiciii.icustay_detail i
      LEFT JOIN mimiciii.icustays s ON i.icustay_id = s.icustay_id
      WHERE s.first_careunit NOT like 'NICU'
      and i.hospstay_seq = 1
      and i.icustay_seq = 1
      and i.age >= 15
      and i.los_icu >= 0.5
      and i.subject_id > %d
      and i.subject_id < %d
      and i.admittime <= i.intime
      and i.intime <= i.outtime
      and i.outtime <= i.dischtime
    ''' % (min_id,max_id)
    first_icu = pd.read_sql_query(first_icu_query, con)

    # You can change the query, but you still gotta pass this constraint
    for i,row in first_icu.iterrows():
        assert row.admittime <= row.intime <= row.outtime <= row.dischtime

    # Query mimic for notes
    notes_query = \
    """
    select n.subject_id,n.hadm_id,n.charttime,n.category,n.text
    from mimiciii.noteevents n
    where iserror IS NULL --this is null in mimic 1.4, rather than empty space
    and subject_id > %d
    and subject_id < %d
    and category != 'Discharge summary'
    and hadm_id IS NOT NULL
    and charttime IS NOT NULL
    ;
    """ % (min_id,max_id)
    notes = pd.read_sql_query(notes_query, con)

    # consider all notes from patient's first hospital admission
    first_hadm_notes = pd.merge(first_icu, notes, on=['subject_id', 'hadm_id'])

    # filter down to only notes that happened during first ICU stay
    first_icu_inds = []
    for i,row in first_hadm_notes.iterrows():
        if row.intime <= row.charttime <= row.outtime:
            first_icu_inds.append(i)
    first_icu_notes = first_hadm_notes.loc[first_icu_inds]

    # only look at these three kinds of notes
    categories = set(['Radiology','Nursing','Physician'])

    # notes data
    text_data = defaultdict(list)
    for i,row in first_icu_notes.iterrows():
        # only looking at the common categories
        category = normalize_category(row.category)
        if category not in categories:
            continue

        # only get first 24 notes
        if len(text_data[row.subject_id]) >= 24:
            continue

        # timestamp of arrival
        time = row.charttime - row.intime
        assert time>=datetime.timedelta(days=0)
        text = tokenize(row.text)
        #text = row.text
        data = (time,category,text)
        text_data[row.subject_id].append(data)
    text_data = dict(text_data)

    # static demographic info
    demographics_query = 'select icustay_id,gender,ethnicity,age from mimiciii.icustay_detail where subject_id>%d and subject_id<%d;' % (min_id,max_id)
    demographics = pd.read_sql_query(demographics_query, con)

    # mortality outcome
    mortality_query = 'select icustay_id,hospital_expire_flag from mimiciii.icustay_detail where subject_id>%d and subject_id<%d;' % (min_id,max_id)
    mortality = pd.read_sql_query(mortality_query, con)

    # Get the SAPS scores
    saps_query = 'select icustay_id,sapsii from mimiciii.sapsii where subject_id>%d and subject_id<%d;' % (min_id,max_id)
    saps = pd.read_sql_query(saps_query, con)

    # icustay info
    stay_query = 'select hadm_id,icustay_id,los,first_careunit,last_careunit,first_wardid,last_wardid from mimiciii.icustays where subject_id>%d and subject_id<%d' % (min_id,max_id)
    stay = pd.read_sql_query(stay_query, con)

    # admissions info
    admissions_query = 'select hadm_id,admission_type,admission_location,discharge_location,insurance,language,marital_status,diagnosis from mimiciii.admissions where subject_id>%d and subject_id<%d;' % (min_id,max_id)
    admissions = pd.read_sql_query(admissions_query, con)

    static = pd.merge(demographics, mortality , on=['icustay_id'])
    static = pd.merge(static      , saps      , on=['icustay_id'])
    static = pd.merge(static      , stay      , on=['icustay_id'])
    static = pd.merge(static      , admissions, on=['hadm_id'])

    # note: iterating over notes => one hadm_id per subject_id
    subject2hadm = {}
    for i,row in first_icu_notes.iterrows():
        subject_id = row.subject_id
        hadm_id = row.hadm_id

        # have to check here in case filtering out some categories removes the only notes
        if subject_id in text_data:
            if subject_id in subject2hadm:
                assert subject2hadm[subject_id] == hadm_id
            else:
                subject2hadm[subject_id] = hadm_id


    def val(item):
        return item.values[0]

    # Many structued clinical variables in mimic
    structured_data = {}
    for subject_id,hadm_id in subject2hadm.items():
        assert subject_id not in structured_data
        static_row = static.loc[static['hadm_id'] == hadm_id]
        info = {
                'subject_id':subject_id,
                'gender'   :val(static_row['gender'])   , 'age'   :val(static_row['age']),
                'ethnicity':val(static_row['ethnicity']), 'sapsii':val(static_row['sapsii']),
                'los'             :val(static_row['los'])      ,
                'first_careunit'  :val(static_row['first_careunit']),
                'last_careunit'   :val(static_row['last_careunit']),
                'first_wardid'    :val(static_row['first_wardid']),
                'last_wardid'     :val(static_row['last_wardid']),
                'hosp_expire_flag':val(static_row['hospital_expire_flag']),
                'admission_type'    :val(static_row['admission_type']),
                'admission_location':val(static_row['admission_location']),
                'discharge_location':val(static_row['discharge_location']),
                'insurance'         :val(static_row['insurance']),
                'language'          :val(static_row['language']),
                'marital_status'    :val(static_row['marital_status']),
                'diagnosis'         :val(static_row['diagnosis']),
               }
        structured_data[subject_id] = info


    return text_data, structured_data



def normalize_category(cat):
    cat = cat.strip()
    if 'Nursing' in cat:
        cat = 'Nursing'
    return cat



def timestamp(tup):
    dt = tup[0]
    if isinstance(dt, pd._libs.tslib.NaTType):
        return None
    else:
        return dt



regex_punctuation  = re.compile('[\',\.\-/\n]')
regex_alphanum     = re.compile('[^a-zA-Z0-9_ ]')
regex_num          = re.compile('\d[\d ]+')
regex_sectionbreak = re.compile('____+')
def tokenize(text):
    text = text.strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    # collapse phrases (including diagnoses) into single tokens
    if text != text.upper():
        caps_matches = set(re.findall('([A-Z][A-Z_ ]+[A-Z])', text))
        for caps_match in caps_matches:
            caps_match = re.sub(' +', ' ', caps_match)
            if len(caps_match) < 35:
                replacement = caps_match.replace(' ','_')
                text = text.replace(caps_match,replacement)

    year_regexes = ['(\d+) years? old', '\s(\d+) ?yo ', '(\d+)[ -]year-old',
                    '\s(\d+) yr old', '\s(\d+) yo[m/f]', '(\d+) y/o ']
    year_text = ' ' + text.lower()
    for year_regex in year_regexes:
        year_matches = re.findall(year_regex, year_text)
        for match in set(year_matches):
            binned_age = ' %s ' % bin_age(match)
            text = text.replace(match, binned_age)

    text = re.sub('_+', '_', text)

    text = text.lower()
    text = re.sub(regex_punctuation , ' '  , text)
    text = re.sub(regex_alphanum    , ''   , text)
    text = re.sub(regex_num         , ' 0 ', text)
    return text.strip().split()


def bin_age(age):
    age = int(age)
    if age < 20:
        return 'AGE_LESS_THAN_TWENTY'
    if age < 30:
        return 'AGE_BETWEEN_TWENTY_AND_THIRTY'
    if age < 40:
        return 'AGE_BETWEEN_THIRTY_AND_FOURTY'
    if age < 50:
        return 'AGE_BETWEEN_FOURTY_AND_FIFTY'
    if age < 60:
        return 'AGE_BETWEEN_FIFTY_AND_SIXTY'
    if age < 70:
        return 'AGE_BETWEEN_SIXTY_AND_SEVENTY'
    if age < 80:
        return 'AGE_BETWEEN_SEVENTY_AND_EIGHTY'
    if age < 90:
        return 'AGE_BETWEEN_EIGHTY_AND_NINETY'
    if age > 90:
        return 'AGE_OVER_NINETY'
    raise Exception('shouldnt get here')



def dump_readable(X, Y):
    for pid in X:
        filename = os.path.join(readable_dir, '%s.txt' % pid)
        with open(filename, 'w') as f:
            if Y[pid]['hosp_expire_flag']:
                print >>f, 'died in hospital'
            else:
                print >>f, 'survived!'
            print >>f, ''

            # count number of reports
            cats = defaultdict(int)
            for dt,c,text in X[pid]:
                cats[c] += 1
            for cat,count in sorted(cats.items()):
                print >>f, '%2d %s' % (count,cat)
            print >>f, ''

            # structured data
            for featurename,value in sorted(Y[pid].items()):
                print >>f, '\t%-20s: %s' % (featurename, value)
            print >>f, ''

            # last note timestamp
            timestamps = [ dt for dt,c,text in X[pid] if not isinstance(dt,pd._libs.tslib.NaTType) ]
            print >>f, ''
            if len(timestamps):
                last = sorted(timestamps)[-1]
            else:
                print X[pid]
                last = X[pid][0][0]
            print >>f, 'last note:', last
            print >>f, ''

            #for i,record in enumerate(sorted(X[pid], key=timestamp)):
            for i,record in enumerate(sorted(X[pid])):
                dt,category,note = record
                print >>f, '='*120
                print >>f, 'note:', i
                print >>f, dt
                print >>f, category
                print >>f, '-'*30
                #print >>f, note.strip()
                print >>f, ' '.join(note).strip()
                print >>f, '*'*120
                print >>f, '\n'



if __name__ == '__main__':
    main()
