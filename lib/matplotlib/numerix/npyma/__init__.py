import warnings

warnings.warn("npyma is obsolete and will be removed", DeprecationWarning)
try:
    from numpy.ma import *        # numpy 1.05 and later
except ImportError:
    from numpy.core.ma import *   # earlier
