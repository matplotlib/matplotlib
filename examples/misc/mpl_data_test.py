"""
Demonstrate how get_mpl_data works with svn revisions in the data.

    svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/mpl_data

and edit testdata.csv to add a new row.  After committing the changes,
when you rerun this script you will get the updated data (and the new
svn version will be cached in ~/.matplotlib/mpl_data)
"""

import matplotlib.cbook as cbook
fh = cbook.get_mpl_data("testdata.csv")
print fh.read()
