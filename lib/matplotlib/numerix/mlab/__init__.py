from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra.mlab import *
elif which[0] == "numeric":
    from MLab import *
else:
    raise RuntimeError("invalid numerix selector")

# for easy access to these functions w/o clobbering builtins
amin = min
amax = max
