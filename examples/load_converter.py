from pylab import figure, show, datestr2num, load
dates, closes = load('data/msft.csv', delimiter=',',
                    converters={0:datestr2num}, skiprows=1, usecols=(0,2),
                    unpack=True)

fig = figure()
ax = fig.add_subplot(111)
ax.plot_date(dates, closes)
show()
