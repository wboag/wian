import cPickle as pickle
import sys
import os
from os.path import dirname
import tempfile
import atexit
import shutil


# find where umls tables are located
homedir = dirname(dirname(dirname(os.path.abspath(__file__))))
umls_tables = os.path.join(homedir, 'resources', 'umls_tables')


class UmlsCache:
    filename = None
    cache = None

    def __init__(self):
        try:
            UmlsCache.filename = os.path.join(umls_tables, 'umls_cache')
            with open(UmlsCache.filename, 'rb') as f:
                UmlsCache.cache = pickle.load(f)
        except IOError:
            UmlsCache.cache = {}

    def __contains__(self, string):
        return string in UmlsCache.cache

    def __setitem__(self, string, mapping):
        UmlsCache.cache[string] = mapping

    def __getitem__(self, string):
        return UmlsCache.cache[string]

    @staticmethod
    @atexit.register
    def destructor():
        # tmp out file in case CNTRL-C interrupts this dump
        _, tmpfile = tempfile.mkstemp(prefix='wboag-', dir='/tmp')
        with open(tmpfile, 'wb') as f:
            pickle.dump(UmlsCache.cache, f)
        shutil.move(tmpfile, UmlsCache.filename)

