import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_cntr import *
    except ImportError:
        numerix._import_fail_message("_contour", "_na")
        raise
elif numerix.which[0] == "numeric":
    try:
        from matplotlib._nc_cntr import *
    except ImportError:
        numerix._import_fail_message("_contour", "_nc")
        raise
else:                 # Must be numpy
    try:
        from matplotlib._ns_cntr import *
    except ImportError:
        numerix._import_fail_message("_contour", "_ns")
        raise



