"""
Demonstrate how get_sample_data works with git revisions in the data.

    git clone git@github.com/matplotlib/sample_data.git

and edit testdata.csv to add a new row.  After committing the changes,
when you rerun this script you will get the updated data (and the new
git version will be cached in ~/.matplotlib/sample_data)
"""

import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

# get the file handle to the cached data and print the contents
datafile = 'testdir/subdir/testsub.csv'
fh = cbook.get_sample_data(datafile)
print fh.read()

# make sure we can read it using csv2rec
fh.seek(0)
r = mlab.csv2rec(fh)

print mlab.rec2txt(r)

fh.close()

