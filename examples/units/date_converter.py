import date_support # set up the date converters
import datetime
from pylab import figure, show, nx


xmin = datetime.date(2007,1,1)
xmax = datetime.date.today()

xdates = [xmin]
while 1:
    thisdate = xdates[-1] + datetime.timedelta(days=1)
    xdates.append(thisdate)
    if thisdate>=xmax: break

fig = figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)
ax.plot(xdates, nx.mlab.rand(len(xdates)), 'o')
ax.set_xlim(datetime.date(2007,2,1), datetime.date(2007,3,1))

for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
show()
