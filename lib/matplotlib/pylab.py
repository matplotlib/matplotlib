"""
`pylab` is a historic interface and its use is strongly discouraged. The equivalent
replacement is `matplotlib.pyplot`. See :ref:`api_interfaces` for a full overview
of Matplotlib interfaces.

`pylab` was designed to support a MATLAB-like way of working with all plotting related
functions directly available in the global namespace. This was achieved through a
wildcard import (``from pylab import *``).

.. warning::
   The use of `pylab` is discouraged for the following reasons:

   ``from pylab import *`` imports all the functions from `matplotlib.pyplot`, `numpy`,
   `numpy.fft`, `numpy.linalg`, and `numpy.random`, and some additional functions into
   the global namespace.

   Such a pattern is considered bad practice in modern python, as it clutters the global
   namespace. Even more severely, in the case of `pylab`, this will overwrite some
   builtin functions (e.g. the builtin `sum` will be replaced by `numpy.sum`), which
   can lead to unexpected behavior.
"""

# Utility functions and classes from Matplotlib
from matplotlib.cbook import flatten, silent_list

# Main Matplotlib library import
import matplotlib as mpl

# Date and time handling functions from Matplotlib
from matplotlib.dates import (
    date2num, num2date, datestr2num, drange, DateFormatter, DateLocator,
    RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator,
    HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR,
    SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY,
    relativedelta
)

# Various utility functions from mlab
from matplotlib.mlab import (
    detrend, detrend_linear, detrend_mean, detrend_none, window_hanning, window_none
)

# Import specific functions from Matplotlib and Pyplot
from matplotlib import cbook, mlab, pyplot as plt

# Import numpy and its submodules with alias to avoid conflicts
import numpy as np
import numpy.ma as ma
import numpy.fft as np_fft
import numpy.random as np_random
import numpy.linalg as np_linalg

# Import datetime to avoid numpy's datetime hiding the standard library's datetime
import datetime

# Override specific numpy functions with their standard library equivalents
builtins = __import__("builtins")
bytes = builtins.bytes  # Override numpy's bytes
abs = builtins.abs  # Override numpy's abs
bool = builtins.bool  # Override numpy's bool
max = builtins.max  # Override numpy's max
min = builtins.min  # Override numpy's min
pow = builtins.pow  # Override numpy's pow
round = builtins.round  # Override numpy's round
