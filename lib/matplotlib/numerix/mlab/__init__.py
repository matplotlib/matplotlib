from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra.mlab import *
elif which[0] == "numeric":
    from MLab import *
elif which[0] == "scipy":
    from scipy.base.mlab import *
else:
    raise RuntimeError("invalid numerix selector")

amin = min
amax = max
