import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_transforms import *
    except ImportError:
        numerix._import_fail_message("_transforms", "_na")
        raise
else:
    try:
        from matplotlib._nc_transforms import *
    except ImportError:
        numerix._import_fail_message("_transforms", "_nc")
        raise
