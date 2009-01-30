from matplotlib import mlab
from pylab import figure, show

a = mlab.csv2rec('../data/msft.csv')
a.sort()
print a.dtype

fig = figure()
ax = fig.add_subplot(111)
ax.plot(a.date, a.adj_close, '-')
fig.autofmt_xdate()

# if you have xlwt installed, you can output excel
import mpl_toolkits.exceltools as exceltools
exceltools.rec2excel(a, 'test.xls')
show()
