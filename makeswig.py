"""
requires swig cvs as of 2005-02-24
"""
import os, sys
wrap = (
    'agg',
    )
#SWIG = 'swig'
SWIG = '/home/titan/johnh/dev/bin/swig'
AGGINCLUDE = 'agg23/include'

swigit = '%(SWIG)s -python -c++ -outdir lib/matplotlib -o src/%(SWIGFILE)s.cxx -I%(AGGINCLUDE)s swig/%(SWIGFILE)s.i '


os.system('%(SWIG)s -python -external-runtime src/swig_runtime.h'%locals())
for SWIGFILE in wrap:
    command = swigit%locals()
    print 'swigging %s'%SWIGFILE
    print command
    os.system(command)



