import sys
from matplotlib.matlab import *
from data_helper import get_daily_data

intc, msft = get_daily_data()

delta1 = diff(intc.open)/intc.open[0]

volume = 0.003*intc.volume[:-2]/intc.volume[0]
close = 0.003*intc.close[:-2]/0.003*intc.open[:-2]
scatter(delta1[:-1], delta1[1:], c=close, s=volume)
set(gca(), 'xticks', arange(-0.06, 0.061, 0.02))
set(gca(), 'yticks', arange(-0.06, 0.061, 0.02))
xlabel('Delta day i')
ylabel('Delta day i+1')
title('Delta price as a function of volume and percent change')
grid(True)
show()


