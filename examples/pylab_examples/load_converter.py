from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.dates as mdates

# Note: matplotlib.dates doesn't have bytespdate2num.
# This function was copied off the internet.
# Source: http://pythonprogramming.net/colors-fills-matplotlib-tutorial/
def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)

dates, closes = np.loadtxt(
    datafile, delimiter=',',
    converters={0: bytespdate2num('%d-%b-%y')},
    skiprows=1, usecols=(0, 2), unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, closes, '-')
fig.autofmt_xdate()
plt.show()
