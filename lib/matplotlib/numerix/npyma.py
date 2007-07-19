from matplotlib.numerix import use_maskedarray

from numpy.core.ma import *

if use_maskedarray:
    from maskedarray import *
    print "using maskedarray"
else:
    from numpy.core.ma import *
    #print "using ma"
