"""
numerix imports numpy with some compatibility adjustments for old
code that had been based on Numeric.

It is deprecated and will go away soon.
"""

import sys, os, struct
from matplotlib import rcParams, verbose

import warnings
msg = """
**********************************************************
matplotlib.numerix and all its subpackages are deprecated.
They will be removed soon.  Please use numpy instead.
**********************************************************
"""
warnings.warn(msg, DeprecationWarning)

which = "numpy", "defaulted"  # This is now the only choice

try:
    import numpy.oldnumeric as numpy
    from numpy.oldnumeric import *
except ImportError:
    import numpy
    from numpy import *
    print 'except asarray', asarray
from _sp_imports import nx, infinity, rand, randn, isnan, all, any
from _sp_imports import UInt8, UInt16, UInt32, Infinity
try:
    from numpy.oldnumeric.matrix import Matrix
except ImportError:
    Matrix = matrix
version = 'numpy %s' % numpy.__version__
from numpy import nan


from mlab import amin, amax
newaxis = NewAxis
from numpy import angle
def typecode(a):
    return a.dtype.char
def iscontiguous(a):
    return a.flags.contiguous
def byteswapped(a):
    return a.byteswap()
def itemsize(a):
    return a.itemsize

verbose.report('numerix %s'%version)
# a bug fix for blas numeric suggested by Fernando Perez
matrixmultiply=dot
asum = sum


def _import_fail_message(module, version):
    """Prints a message when the array package specific version of an extension
    fails to import correctly.
    """
    _dict = { "which" : which[0],
              "module" : module,
              "specific" : version + module
              }
    print """
The import of the %(which)s version of the %(module)s module,
%(specific)s, failed.  This is is either because %(which)s was
unavailable when matplotlib was compiled, because a dependency of
%(specific)s could not be satisfied, or because the build flag for
this module was turned off in setup.py.  If it appears that
%(specific)s was not built, make sure you have a working copy of
%(which)s and then re-install matplotlib. Otherwise, the following
traceback gives more details:\n""" % _dict

g = globals()
l = locals()
__import__('ma', g, l)
__import__('fft', g, l)
__import__('linear_algebra', g, l)
__import__('random_array', g, l)
__import__('mlab', g, l)

la = linear_algebra
ra = random_array
