"""numerix  imports either Numeric or numarray based on various selectors.

0.  If the value "--numarray" or "--Numeric" is specified on the
command line, then numerix imports the specified array package.

1. The value of numerix in ~/.matplotlibrc: either Numeric or numarray

2. If none of the above is done, the default array package is Numeric.
Because the .matplotlibrc always provides *some* value for numerix (it
has it's own system of default values), this default is most likely
never used.

3. Whichever numerical package is selected, if the import fails, the
other is tried.

To summarize: the  commandline is examined first, the  rc file second,
and the default array package is Numeric.  If the selected package
fails to import, the other is tried.
"""

import sys, os
from matplotlib import rcParams

which = None, None

# First, see if --numarray or --Numeric was specified on the command
# line:
if hasattr(sys, 'argv'):        #Once again, Apache mod_python has no argv
    for a in sys.argv:
        if a in ["--Numeric", "--numeric", "--NUMERIC",
                 "--Numarray", "--numarray", "--NUMARRAY"]:
            which = a[2:], "command line"
            break

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
    print >>sys.stderr, __doc__
    raise ValueError("numerix selector must be either 'Numeric' or 'numarray' but the value obtained from the %s was '%s'." % (which[1], which[0]))

if which[0] == "numarray":
    try:
        from na_imports import *
    except ImportError:
        print >>sys.stderr, "numarray import failed... trying Numeric."
        which = "numeric", "numarray import error"
        from nc_imports import *
elif which[0] == "numeric":
    try:
        from nc_imports import *
    except ImportError:  
        if which[1] != "defaulted":
            print >>sys.stderr, "Numeric import failed... trying numarray."
            which = "numarray", "Numeric import error"
        else:
            which = "numarray", "defaulted"
        from na_imports import *
else:
    raise RuntimeError("invalid numerix selector")
