import sqlite3
import os
import string
from os.path import dirname

import create_sqliteDB
import cache


# find where umls tables are located
homedir = dirname(dirname(dirname(os.path.abspath(__file__))))
umls_tables = os.path.join(homedir, 'resources', 'umls_tables')



############################################
###          Setups / Handshakes         ###
############################################


#connect to UMLS database
def SQLConnect():
    #try to connect to the sqlite database.
    # if database does not exit. Make one.
    db_path = os.path.join(umls_tables, "umls.db")
    if not os.path.isfile(db_path):
        print "\n\tdb doesn't exist (creating one now)\n"
        create_sqliteDB.create_db()

    db = sqlite3.connect(db_path)
    return db.cursor()




############################################
###      Global reource connections      ###
############################################


# Global database connection
conn = SQLConnect()

# cache for UMLS queries
C = cache.UmlsCache()



############################################
###           Query Operations           ###
############################################


def str_lookup(string):
    """ Get sty for a given string """
    key = ('str_lookup',string)
    if key in C:
        return C[key]
    try:
        conn.execute( "SELECT sty FROM MRCON a, MRSTY b WHERE a.cui = b.cui AND str = ?; " , (string,) )
        results = conn.fetchall()
    except sqlite3.ProgrammingError, e:
        results = []
    #C[key] = results
    return results


def cui_lookup(string):
    """ get cui for a given string """
    key = ('cui_lookup',string)
    if key in C:
        return C[key]
    try:
        # Get cuis
        conn.execute( "SELECT cui FROM MRCON WHERE str = ?;" , (string,) )
        results = conn.fetchall()
    except sqlite3.ProgrammingError, e:
        results = []
    #C[key] = results
    return results


def abr_lookup(string):
    """ searches for an abbreviation and returns possible expansions for that abbreviation"""
    key = ('abr_lookup',string)
    if key in C:
        return C[key]
    try:
        conn.execute( "SELECT str FROM LRABR WHERE abr = ?;", (string,))
        results = conn.fetchall()
    except sqlite3.ProgrammingError, e:
        results = []
    #C[key] = results
    return results



def tui_lookup(string):
    """ takes in a concept id string (ex: C00342143) and returns the TUI of that string which represents the semantic type is belongs to """
    key = ('tui_lookup',string)
    if key in C:
        return C[key]
    try:
        conn.execute( "SELECT tui FROM MRSTY WHERE cui = ?;", (string,))
        results = conn.fetchall()
    except sqlite3.ProgrammingError, e:
        results = []
    #C[key] = results
    return results



def strip_punct(stringArg):
    for c in string.punctuation:
        stringArg = string.replace(stringArg, c, "")
    return stringArg



if __name__ == '__main__':
    print "str_lookup('blood'):", str_lookup('blood')
    print "cui_lookup('blood'):", cui_lookup('blood')
    print "abr_lookup('blood'):", abr_lookup('p.o.')
    print "tui_lookup('blood'):", tui_lookup('C0005767')

