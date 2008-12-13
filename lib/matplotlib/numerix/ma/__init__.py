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
    try:
        from numpy.ma import *        # numpy 1.05 and later
    except ImportError:
        from numpy.core.ma import *   # earlier
    def getmaskorNone(obj):
        _msk = getmask(obj)
        if _msk is nomask:
            return None
        return _msk
else:
    raise RuntimeError("invalid numerix selector")
