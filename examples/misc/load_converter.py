"""
==============
Load Converter
==============

"""

import dateutil.parser
from matplotlib import cbook, dates
import matplotlib.pyplot as plt
import numpy as np


datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)

data = np.genfromtxt(
    datafile, delimiter=',', names=True,
    converters={0: lambda s: dates.date2num(dateutil.parser.parse(s))})

fig, ax = plt.subplots()
ax.plot_date(data['Date'], data['High'], '-')
fig.autofmt_xdate()
plt.show()
