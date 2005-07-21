"""numerix  imports either Numeric or numarray based on various selectors.

0.  If the value "--numarray" or "--Numeric" is specified on the
command line, then numerix imports the specified array package.

1. The value of numerix in ~/.matplotlibrc: either Numeric or numarray

2. If none of the above is done, the default array package is Numeric.
Because the .matplotlibrc always provides *some* value for numerix (it
has it's own system of default values), this default is most likely
never used.

To summarize: the  commandline is examined first, the  rc file second,
and the default array package is Numeric.  
"""

import sys, os
from matplotlib import rcParams, verbose

which = None, None

# First, see if --numarray or --Numeric was specified on the command
# line:

for a in sys.argv:
    if a in ["--Numeric", "--numeric", "--NUMERIC",
             "--Numarray", "--numarray", "--NUMARRAY"]:
        which = a[2:], "command line"
        break
    del a

if which[0] is None:     
    try:  # In theory, rcParams always has *some* value for numerix.
        which = rcParams['numerix'], "rc"
    except KeyError:
        pass

# If all the above fail, default to Numeric. Most likely not used.
if which[0] is None:
    which = "numeric", "defaulted"

which = which[0].strip().lower(), which[1]
if which[0] not in ["numeric", "numarray"]:
    raise ValueError("numerix selector must be either 'Numeric' or 'numarray' but the value obtained from the %s was '%s'." % (which[1], which[0]))

if which[0] == "numarray":
    #from na_imports import *
    from numarray import *
    from _na_imports import nx, inf, infinity, Infinity, Matrix
    from numarray.numeric import nonzero
    from numarray.convolve import cross_correlate, convolve
    import numarray
    
    version = 'numarray %s'%numarray.__version__

elif which[0] == "numeric":
    #from nc_imports import *
    from Numeric import *
    from _nc_imports import nx, inf, infinity, Infinity
    from Matrix import Matrix
    import Numeric
    version = 'Numeric %s'%Numeric.__version__
else:
    raise RuntimeError("invalid numerix selector")

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
