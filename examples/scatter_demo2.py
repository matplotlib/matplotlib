#!/usr/bin/env python
import sys
from matplotlib.matlab import *
from data_helper import get_daily_data

intc, msft = get_daily_data()

delta1 = diff(intc.open)/intc.open[0]

volume = (10*intc.volume[:-2]/intc.volume[0])**2
close = 0.003*intc.close[:-2]/0.003*intc.open[:-2]
p = scatter(delta1[:-1], delta1[1:], c=close, s=volume)
set(p, 'alpha', 0.75)
set(gca(), 'xticks', arange(-0.06, 0.061, 0.02))
set(gca(), 'yticks', arange(-0.06, 0.061, 0.02))
xlabel(r'$\Delta_i$', fontsize='x-large')
ylabel(r'$\Delta_{i+1}$', fontsize='x-large')
title(r'Volume and percent change')
grid(True)
#savefig('scatter_demo2')
show()


