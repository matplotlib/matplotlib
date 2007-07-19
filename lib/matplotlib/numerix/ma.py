from matplotlib.numerix import use_maskedarray

from numpy.core.ma import *

if use_maskedarray:
    from maskedarray import *
    print "using maskedarray"
else:
    from numpy.core.ma import *
    #print "using ma"

def getmaskorNone(obj):
    _msk = getmask(obj)
    if _msk is nomask:
        return None
    return _msk
