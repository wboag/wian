
import os, sys
import psycopg2
import pandas as pd
import cPickle as pickle
from collections import defaultdict


homedir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
datadir = os.path.join(homedir, 'data')
if not os.path.exists(datadir):
    os.makedirs(datadir)


def main():

    try:
        mode = sys.argv[1]
        if mode not in ['all','small']:
            raise Exception('bad')
    except Exception, e:
        print '\n\tusage: python %s <all|small>\n' % sys.argv[0]
        exit(1)


    # this is a speedup during development
    if mode == 'small':
        max_id = 300
    elif mode == 'all':
        max_id = 1e20
    else:
        raise Exception('bad mode "%s"' % mode)


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
      and i.subject_id < %d
      and i.admittime <= i.intime
      and i.intime <= i.outtime
      and i.outtime <= i.dischtime
    ''' % (max_id)
    first_icu = pd.read_sql_query(first_icu_query, con)
    icu_ids = set(first_icu['subject_id'].values)

    # You can change the query, but you still gotta pass this constraint
    for i,row in first_icu.iterrows():
        assert row.admittime <= row.intime <= row.outtime <= row.dischtime

    # dialysis
    rrt_query = 'select subject_id,icustay_id,rrt from mimiciii.rrt where subject_id<%d;' % max_id
    rrt = pd.read_sql_query(rrt_query, con)

    # was dialysis given during the final icu stay?
    dialysis = {}
    for ind,row in pd.merge(first_icu,rrt,on=['subject_id','icustay_id']).iterrows():
        dialysis[row.subject_id] = row.rrt


    # Marzyeh data has ventilation and vasopressors
    treatments = pd.read_hdf('/scratch/mghassem/phys_acuity_modelling/data/Y.h5')

    # query Marzyeh's data (filter down to something smaller)
    vent = defaultdict(list)
    vaso = defaultdict(list)
    unique_subject_icu = {}
    for index,row in treatments.iterrows():
        subject_id,hadm_id,icustay_id,hours_in = index

        # filter out anyone who did not die in hospital
        if subject_id not in icu_ids:
            continue

        # ensure each icu only has one subject_id
        if subject_id in unique_subject_icu:
            assert unique_subject_icu[subject_id] == icustay_id
        else:
            unique_subject_icu[subject_id] = icustay_id

        # ensure traversing list in order
        assert len(vent[subject_id]) == hours_in
        vent[subject_id].append(row.vent)
        vaso[subject_id].append(row.vaso)


    # Nathan data has boluses
    bolus_treatments = pd.read_hdf('/scratch/nhunt/phys_acuity_modelling/data/Y.h5')

    # query Marzyeh's data (filter down to something smaller)
    col = defaultdict(list)
    crys = defaultdict(list)
    unique_subject_icu = {}
    for index,row in bolus_treatments.iterrows():
        subject_id,hadm_id,icustay_id,hours_in = index

        # filter out anyone who did not die in hospital
        if subject_id not in icu_ids:
            continue

        # ensure each icu only has one subject_id
        if subject_id in unique_subject_icu:
            assert unique_subject_icu[subject_id] == icustay_id
        else:
            unique_subject_icu[subject_id] = icustay_id

        # ensure traversing list in order
        assert len(col[subject_id]) == hours_in
        col[subject_id].append(row.colloid_bolus)
        crys[subject_id].append(row.crystalloid_bolus)

    vaso = { pid:int(any(t)) for pid,t in vaso.items() }
    vent = { pid:int(any(t)) for pid,t in vent.items() }
    col  = { pid:int(any(t)) for pid,t in  col.items() }
    crys = { pid:int(any(t)) for pid,t in crys.items() }

    # putting it all together
    eol_treatments = {'dialysis':dialysis, 'vaso':dict(vaso), 'vent':dict(vent), 'col':dict(col), 'crys':dict(crys)}

    # save this data to file
    filename = os.path.join(datadir, '%s_treatments.pickle' % mode)
    print 'serializing treatments:', filename
    with open(filename, 'wb') as f:
        pickle.dump(eol_treatments, f)



if __name__ == '__main__':
    main()
