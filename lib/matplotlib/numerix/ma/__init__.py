from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.ma import *
elif which[0] == "numeric":
    from MA import *
else:
    raise RuntimeError("invalid numerix selector")

