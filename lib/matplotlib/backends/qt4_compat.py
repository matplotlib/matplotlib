import warnings
from matplotlib.cbook import mplDeprecation
# bulk-imports because we are pretending that file is this file
from .qt_compat import *
_warn_str = ("This module has been deprecated in 1.4 in favor "
             "of matplotlib.backends.qt_compat\n"
             "This module will be removed in 1.5, please update "
             "your imports.")

warnings.warn(_warn_str, mplDeprecation)
