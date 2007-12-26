from matplotlib.numerix import use_maskedarray

if use_maskedarray:
    from maskedarray import *
    print "using maskedarray"
else:
    try:
        from numpy.ma import *        # numpy 1.05 and later
    except ImportError:
        from numpy.core.ma import *   # earlier
    #print "using ma"
