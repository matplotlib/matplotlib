"""
==============
Load converter
==============

This example demonstrates passing a custom converter to `numpy.genfromtxt` to
extract dates from a CSV file.
"""

import dateutil.parser
from matplotlib import cbook
import matplotlib.pyplot as plt
import numpy as np


datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)

data = np.genfromtxt(
    datafile, delimiter=',', names=True,
    dtype=None, converters={0: dateutil.parser.parse})

fig, ax = plt.subplots()
ax.plot(data['Date'], data['High'], '-')
fig.autofmt_xdate()
plt.show()
