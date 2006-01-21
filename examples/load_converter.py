from pylab import figure, show, datestr2num, load



X = load('data/msft.csv', delimiter=',',
         converters={0:datestr2num}, skiprows=1)
dates = X[:,0]
close = X[:,2]

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, close)
show()
