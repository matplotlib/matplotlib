import sys
from matplotlib.matlab import *
from data_helper import get_two_stock_data
import matplotlib.numerix as numpy
(d1, p1, d2, p2 ) = get_two_stock_data()

lines = plot(d1, p1, 'bs', d2, p2, 'go')
set(lines, 'data_clipping', True)
xlabel('Days')
ylabel('Normalized price')
set(gca(), 'xlim', [0, 3])
title('INTC vs AAPL')
legend( ('INTC', 'AAPL') )
#savefig('stock_demo')
show()


