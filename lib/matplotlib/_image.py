import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_image import *
    except ImportError:
        numerix._import_fail_message("_image", "_na")
        raise
else:
    try:
        from matplotlib._nc_image import *
    except ImportError:
        numerix._import_fail_message("_image", "_nc")
        raise
