from matplotlib.mlab import csv2rec
from pylab import figure, show

a = csv2rec('data/msft.csv')
print a.dtype

fig = figure()
ax = fig.add_subplot(111)
ax.plot(a.date, a.adj_close, '-')
fig.autofmt_xdate()
show()
