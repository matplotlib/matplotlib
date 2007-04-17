from matplotlib.dates import strpdate2num
from matplotlib.mlab import load
from pylab import figure, show

dates, closes = load(
    'data/msft.csv', delimiter=',',
    converters={0:strpdate2num('%d-%b-%y')},
    skiprows=1, usecols=(0,2), unpack=True)

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, closes, '-')
fig.autofmt_xdate()
show()
