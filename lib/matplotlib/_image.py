import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_image import *
    except ImportError:
        numerix._import_fail_message("_image", "_na")
        raise
elif numerix.which[0] == "numeric":
    try:
        from matplotlib._nc_image import *
    except ImportError:
        numerix._import_fail_message("_image", "_nc")
        raise
else:                 # Must be numpy
    try:
        from matplotlib._ns_image import *
    except ImportError:
        numerix._import_fail_message("_image", "_ns")
        raise

