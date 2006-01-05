import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_transforms import *
    except ImportError:
        numerix._import_fail_message("_transforms", "_na")
        raise
elif numerix.which[0] == "numeric":
    try:
        from matplotlib._nc_transforms import *
    except ImportError:
        numerix._import_fail_message("_transforms", "_nc")
        raise
else:                 # Must be numpy
    try:
        from matplotlib._ns_transforms import *
    except ImportError:
        numerix._import_fail_message("_transforms", "_ns")
        raise



