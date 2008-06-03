#!/usr/bin/env python
import sys
from pylab import *
from data_helper import get_daily_data

intc, msft = get_daily_data()

delta1 = diff(intc.open)/intc.open[0]

volume = (15*intc.volume[:-2]/intc.volume[0])**2
close = 0.003*intc.close[:-2]/0.003*intc.open[:-2]
p = scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.75)

xlabel(r'$\Delta_i$', size='x-large')
ylabel(r'$\Delta_{i+1}$', size='x-large')
title(r'Volume and percent change')
grid(True)
#savefig('scatter_demo2')
colorbar()
show()


