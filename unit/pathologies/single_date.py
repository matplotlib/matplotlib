import time
from matplotlib.dates import EpochConverter
from pylab import *
from matplotlib.ticker import FuncFormatter, NullLocator,\
     MinuteLocator, DayLocator, HourLocator, MultipleLocator, DateFormatter

time1=[1087192789.89]
data1=[-65.54]

p1=plot_date(time1, data1, None, 'o', color='r')
show()

