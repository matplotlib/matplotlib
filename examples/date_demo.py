"""
Simple example showing how to plot a time series with datetime objects
"""
import datetime
import matplotlib.pyplot as plt

today = datetime.date.today()
dates = [today+datetime.timedelta(days=i) for i in range(10)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dates, range(10))
fig.autofmt_xdate()
plt.show()
