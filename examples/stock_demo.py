import sys
from matplotlib.matlab import *
from data_helper import get_two_stock_data
import Numeric as numpy
(d1, p1, d2, p2 ) = get_two_stock_data()

plot(d1, p1, 'bs', d2, p2, 'go')
legend( ('INTC', 'AAPL'))

xlabel('Days')
ylabel('Normalized price')
set(gca(), 'xlim', [0, 3])
title('INTC vs AAPL')
show()


