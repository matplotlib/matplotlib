from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra import *
elif which[0] == "numeric":
    from LinearAlgebra import *
elif which[0] == "numpy":
    try:
        from numpy.oldnumeric.linear_algebra import *
    except ImportError:
        from numpy.linalg.old import *
else:
    raise RuntimeError("invalid numerix selector")
