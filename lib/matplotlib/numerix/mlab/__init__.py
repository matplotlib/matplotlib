from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra.mlab import *
elif which[0] == "numeric":
    from MLab import *
elif which[0] == "numpy":
    try:
       from numpy.oldnumeric.mlab import *
    except ImportError:
       from numpy.lib.mlab import *
else:
    raise RuntimeError("invalid numerix selector")

amin = min
amax = max
