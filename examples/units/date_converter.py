import date_support # set up the date converters
import datetime
from matplotlib.dates import drange
from pylab import figure, show, nx


xmin = datetime.date(2007,1,1)
xmax = datetime.date.today()
delta = datetime.timedelta(days=1)
xdates = drange(xmin, xmax, delta)

fig = figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)
ax.plot(xdates, nx.mlab.rand(len(xdates)), 'o')
ax.set_xlim(datetime.date(2007,2,1), datetime.date(2007,3,1))

fig.autofmt_xdate()
show()
