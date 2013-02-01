import warnings
from matplotlib.utils._cbook import *

from matplotlib import MatplotlibDeprecationWarning as mplDeprecation

warnings.warn("The module cbook is deprecated. It will be removed in 1.5.x. "
              "For more information, please see CHANGELOG files or XXX",
              mplDeprecation)
