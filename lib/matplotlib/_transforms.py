import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_transforms import *
    except ImportError:
        print "numarray wasn't available at compile time so the numarray"
        print "extensions were not built.  Please make sure you have a working"
        print "numarray installation and then rebuild matplotlib."
        sys.exit(1)
else:
    try:
        from matplotlib._nc_transforms import *
    except ImportError:
        print "Numeric wasn't available at compile time so the Numeric"
        print "extensions were not built.  Please make sure you have a working"
        print "Numeric installation and then rebuild matplotlib."
        sys.exit(1)
