import sys

use_maskedarray = None

for a in sys.argv:
    if a == "--maskedarray":
        use_maskedarray = True
    if a == "--ma":
        use_maskedarray = False
del a

if use_maskedarray is None:
    import matplotlib
    try:
        use_maskedarray = matplotlib.rcParams['maskedarray']
    except KeyError:
        use_maskedarray = False

#########################

from numpy import *

#########################

# the following is exclusively used and/or reexported by pylab.py and mlab.py:

asum = sum
matrixmultiply = dot

from numpy.oldnumeric import \
    ArrayType, \
    cross_correlate, \
    arrayrange, \
    innerproduct, \
    outerproduct

from numpy.oldnumeric import Int8, UInt8, \
     Int16, UInt16, \
     Int32, UInt32, \
     Float32, Float64, \
     Complex32, Complex64, \
     Float, Int, Complex

from numpy.oldnumeric.matrix import Matrix

from numpy.oldnumeric.mlab import min as amin
from numpy.oldnumeric.mlab import max as amax

def typecode(a):
    return a.dtype.char
def iscontiguous(a):
    return a.flags.contiguous
