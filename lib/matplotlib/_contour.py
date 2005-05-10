import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_cntr import *
    except ImportError:
        numerix._import_fail_message("_contour", "_na")
        raise
else:
    try:
        from matplotlib._nc_cntr import *
    except ImportError:
        numerix._import_fail_message("_contour", "_nc")
        raise
