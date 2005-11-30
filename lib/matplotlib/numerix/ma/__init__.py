from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.ma import *
elif which[0] == "numeric":
    from MA import *
elif which[0] == "scipy":
    from scipy.base.ma import *
else:
    raise RuntimeError("invalid numerix selector")
