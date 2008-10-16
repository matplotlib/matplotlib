"""
generate the rst files for the examples by iterating over the pylab examples
"""
import os, glob

import matplotlib.cbook as cbook


import os
import sys
fileList = []
rootdir = '../mpl_examples'

datad = {}
for root, subFolders, files in os.walk(rootdir):
    for fname in files:
        if ( fname.startswith('.') or fname.startswith('#') or 
             fname.find('.svn')>=0 or not fname.endswith('.py') ): 
            continue
        
        fullpath = os.path.join(root,fname)
        contents = file(fullpath).read()
        # indent
        contents = '\n'.join(['    %s'%row.rstrip() for row in contents.split('\n')])
        relpath = os.path.split(root)[-1]
        datad.setdefault(relpath, []).append((fname, contents))

subdirs = datad.keys()
subdirs.sort()

fhindex = file('index.rst')
fh.index.write("""\
.. _examples-index:

####################
Matplotlib Examples
###################

.. htmlonly::

    :Release: |version|
    :Date: |today|

.. toctree::
    :maxdepth: 2

""")

for subdir in subdirs:
    print subdir
    outfile = '%s.rst'%subdir
    fh = file(outfile, 'w')

    fhindex.write('    %s\n'%outfile)

    fh.write('.. _%s-examples:\n\n'%subdir)
    title = '%s examples'%subdir

    fh.write('*'*len(title) + '\n')
    fh.write(title + '\n')
    fh.write('*'*len(title) + '\n\n')
    
    for fname, contents in datad[subdir]:
        print '    ', fname
        basename, ext = os.path.splitext(fname)
        fh.write('.. _%s-example:\n\n'%basename)
        title = '%s example'%basename

        fh.write(title + '\n')
        fh.write('='*len(title) + '\n\n')
        fh.write(fname + '::\n\n')
        fh.write(contents)
        fh.write('\n\n')
    fh.close()

     

fhindex.close()
