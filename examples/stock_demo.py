#!/usr/bin/env python
import sys
from matplotlib.ticker import MultipleLocator
from pylab import *
from data_helper import get_two_stock_data
import matplotlib.numerix as numpy
(d1, p1, d2, p2 ) = get_two_stock_data()

ax = subplot(111)
lines = plot(d1, p1, 'bs', d2, p2, 'go')
set(lines, 'data_clipping', True)
xlabel('Days')
ylabel('Normalized price')
xlim( 0, 3) 
ax.xaxis.set_major_locator(MultipleLocator(1))

title('INTC vs AAPL')
legend( ('INTC', 'AAPL') )
#savefig('stock_demo')
show()


