try:
    from numpy.ma import *        # numpy 1.05 and later
except ImportError:
    from numpy.core.ma import *   # earlier
def getmaskorNone(obj):
    _msk = getmask(obj)
    if _msk is nomask:
        return None
    return _msk
