"""
=========================
Plotting data from a file
=========================

Plotting data from a file is actually a two-step process.

1. Interpreting the file and loading the data.
2. Creating the actual plot.

`.pyplot.plotfile` tried to do both at once. But each of the steps has so many
possible variations and parameters that it does not make sense to squeeze both
into a single function. Therefore, `.pyplot.plotfile` has been deprecated.

The recommended way of plotting data from a file is therefore to use dedicated
functions such as `numpy.loadtxt` or `pandas.read_csv` to read the data. These
are more powerful and faster. Then plot the obtained data using matplotlib.

Note that `pandas.DataFrame.plot` is a convenient wrapper around Matplotlib
to create simple plots.
"""

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd

###############################################################################
# Using pandas
# ============
#
# Subsequent are a few examples of how to replace `~.pyplot.plotfile` with
# `pandas`. All examples need the the `pandas.read_csv` call first. Note that
# you can use the filename directly as a parameter::
#
#     msft = pd.read_csv('msft.csv')
#
# The following slightly more involved `pandas.read_csv` call is only to make
# automatic rendering of the example work:

fname = cbook.get_sample_data('msft.csv', asfileobj=False)
with cbook.get_sample_data('msft.csv') as file:
    msft = pd.read_csv(file)

###############################################################################
# When working with dates, additionally call
# `pandas.plotting.register_matplotlib_converters` and use the ``parse_dates``
# argument of `pandas.read_csv`::

pd.plotting.register_matplotlib_converters()

with cbook.get_sample_data('msft.csv') as file:
    msft = pd.read_csv(file, parse_dates=['Date'])


###############################################################################
# Use indices
# -----------

# Deprecated:
plt.plotfile(fname, (0, 5, 6))

# Use instead:
msft.plot(0, [5, 6], subplots=True)

###############################################################################
# Use names
# ---------

# Deprecated:
plt.plotfile(fname, ('date', 'volume', 'adj_close'))

# Use instead:
msft.plot("Date", ["Volume", "Adj. Close*"], subplots=True)

###############################################################################
# Use semilogy for volume
# -----------------------

# Deprecated:
plt.plotfile(fname, ('date', 'volume', 'adj_close'),
             plotfuncs={'volume': 'semilogy'})

# Use instead:
fig, axs = plt.subplots(2, sharex=True)
msft.plot("Date", "Volume", ax=axs[0], logy=True)
msft.plot("Date", "Adj. Close*", ax=axs[1])


###############################################################################
# Use semilogy for volume (by index)
# ----------------------------------

# Deprecated:
plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'semilogy'})

# Use instead:
fig, axs = plt.subplots(2, sharex=True)
msft.plot(0, 5, ax=axs[0], logy=True)
msft.plot(0, 6, ax=axs[1])

###############################################################################
# Single subplot
# --------------

# Deprecated:
plt.plotfile(fname, ('date', 'open', 'high', 'low', 'close'), subplots=False)

# Use instead:
msft.plot("Date", ["Open", "High", "Low", "Close"])

###############################################################################
# Use bar for volume
# ------------------

# Deprecated:
plt.plotfile(fname, (0, 5, 6), plotfuncs={5: "bar"})

# Use instead:
fig, axs = plt.subplots(2, sharex=True)
axs[0].bar(msft.iloc[:, 0], msft.iloc[:, 5])
axs[1].plot(msft.iloc[:, 0], msft.iloc[:, 6])
fig.autofmt_xdate()

###############################################################################
# Using numpy
# ===========

fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)
with cbook.get_sample_data('data_x_x2_x3.csv') as file:
    array = np.loadtxt(file)

###############################################################################
# Labeling, if no names in csv-file
# ---------------------------------

# Deprecated:
plt.plotfile(fname2, cols=(0, 1, 2), delimiter=' ',
             names=['$x$', '$f(x)=x^2$', '$f(x)=x^3$'])

# Use instead:
fig, axs = plt.subplots(2, sharex=True)
axs[0].plot(array[:, 0], array[:, 1])
axs[0].set(ylabel='$f(x)=x^2$')
axs[1].plot(array[:, 0], array[:, 2])
axs[1].set(xlabel='$x$', ylabel='$f(x)=x^3$')

###############################################################################
# More than one file per figure
# -----------------------------

# For simplicity of the example we reuse the same file.
# In general they will be different.
fname3 = fname2

# Depreacted:
plt.plotfile(fname2, cols=(0, 1), delimiter=' ')
plt.plotfile(fname3, cols=(0, 2), delimiter=' ',
             newfig=False)  # use current figure
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x) = x^2, x^3$')

# Use instead:
fig, ax = plt.subplots()
ax.plot(array[:, 0], array[:, 1])
ax.plot(array[:, 0], array[:, 2])
ax.set(xlabel='$x$', ylabel='$f(x)=x^3$')

plt.show()
