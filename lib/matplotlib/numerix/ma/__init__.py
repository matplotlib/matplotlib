from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.ma import *
elif which[0] == "numeric":
    from MA import *
elif which[0] == "numpy":
    from numpy.core.ma import *
else:
    raise RuntimeError("invalid numerix selector")
