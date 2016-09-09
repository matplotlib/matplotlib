"""
Shim to provide backword compatibility for finance
"""
import warnings
from matplotlib.cbook import mplDeprecation
try:
    from mpl_finance import *
except ImportError:
    raise ImportError("Matplotlib.finance has been split into a separate "
                      "module. Please install 'mpl_finance' to use.")
else:
    msg = ("Matplotlib.finance has been split into a separate "
           "module. Please update your imports to "
           "replace 'matplotlib.finance' with 'mpl_finance'. "
           "This shim will be removed in a future release.")
    warnings.warn(msg, mplDeprecation, stacklevel=1)
