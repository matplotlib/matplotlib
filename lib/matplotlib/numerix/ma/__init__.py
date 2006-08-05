from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.ma import *
    nomask = None
    getmaskorNone = getmask
elif which[0] == "numeric":
    from MA import *
    nomask = None
    getmaskorNone = getmask
elif which[0] == "numpy":
    from numpy.core.ma import *
    def getmaskorNone(obj):
        _msk = getmask(obj)
        if _msk is nomask:
            return None
        return _msk
else:
    raise RuntimeError("invalid numerix selector")
