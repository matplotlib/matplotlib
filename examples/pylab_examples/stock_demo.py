"""
================
Stock Demo Plots
================

The following example displays Matplotlib's capabilities of creating
graphs that can be used for stocks. The example specifically uses
Apple and Intel stock data and graphs the normalized prices on the
same plot.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.ticker import MultipleLocator


def get_two_stock_data():
    """
    load stock time and price data for two stocks The return values
    (d1,p1,d2,p2) are the trade time (in days) and prices for stocks 1
    and 2 (intc and aapl)
    """
    file1 = cbook.get_sample_data('INTC.dat.gz')
    file2 = cbook.get_sample_data('AAPL.dat.gz')
    M1 = np.fromstring(file1.read(), '<d')

    M1 = np.resize(M1, (M1.shape[0]//2, 2))

    M2 = np.fromstring(file2.read(), '<d')
    M2 = np.resize(M2, (M2.shape[0]//2, 2))

    d1, p1 = M1[:, 0], M1[:, 1]
    d2, p2 = M2[:, 0], M2[:, 1]
    return (d1, p1, d2, p2)


d1, p1, d2, p2 = get_two_stock_data()

fig, ax = plt.subplots()
lines1 = ax.plot(d1, p1, label="INTC")
lines2 = ax.plot(d2, p2, label="AAPL")
ax.set_xlabel('Days')
ax.set_ylabel('Normalized price')
ax.set_xlim(0, 3)
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.set_title('INTC vs AAPL')
ax.legend()

plt.show()
