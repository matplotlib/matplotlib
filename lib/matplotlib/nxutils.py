import sys, numerix

if numerix.which[0] == "numarray":
    try:
        from matplotlib._na_nxutils import *
    except ImportError:
        numerix._import_fail_message("nxutils", "_na")
        raise
elif numerix.which[0] == "numeric":
    try:
        from matplotlib._nc_nxutils import *
    except ImportError:
        numerix._import_fail_message("nxutils", "_nc")
        raise
else:                 # Must be numpy
    try:
        from matplotlib._ns_nxutils import *
    except ImportError:
        numerix._import_fail_message("nxutils", "_ns")
        raise



