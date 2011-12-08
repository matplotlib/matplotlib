from __future__ import print_function
"""
report how many days it has been since each developer committed.  You
must do an

svn log > log.txt

and place the output next to this file before running

"""
import os, datetime

import matplotlib.cbook as cbook

todate = cbook.todate('%Y-%m-%d')
today = datetime.date.today()
if not os.path.exists('log.txt'):
    print('You must place the "svn log" output into a file "log.txt"')
    raise SystemExit

parse = False

lastd = dict()
for line in file('log.txt'):
    if line.startswith('--------'):
        parse = True
        continue

    if parse:
        parts = [part.strip() for part in line.split('|')]
        developer = parts[1]
        dateparts = parts[2].split(' ')
        ymd = todate(dateparts[0])


    if developer not in lastd:
        lastd[developer] = ymd

    parse = False

dsu = [((today - lastdate).days, developer) for developer, lastdate in lastd.items()]

dsu.sort()
for timedelta, developer in dsu:
    print('%s : %d'%(developer, timedelta))
