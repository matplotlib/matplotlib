"""
=============
Custom Ticker
=============

The :mod:`matplotlib.ticker` module defines many preset tickers, but was
primarily designed for extensibility, i.e., to support user customized ticking.

In this example, a user defined function is used to format the ticks in
millions of dollars on the y-axis.
"""

import matplotlib.pyplot as plt


def millions(x, pos):
    """The two arguments are the value and tick position."""
    return f'${x*1e-6:1.1f}M'


fig, ax = plt.subplots()
# set_major_formatter internally creates a FuncFormatter from the callable.
ax.yaxis.set_major_formatter(millions)
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]
ax.bar(['Bill', 'Fred', 'Mary', 'Sue'], money)
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axis.Axis.set_major_formatter`
