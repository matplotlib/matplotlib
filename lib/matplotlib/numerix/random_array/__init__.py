from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.random_array import *
elif which[0] == "numeric":
    from RandomArray import *
else:
    raise RuntimeError("invalid numerix selector")
