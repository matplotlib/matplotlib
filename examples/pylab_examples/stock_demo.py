import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook
from matplotlib.ticker import MultipleLocator

def get_two_stock_data():
    """
    load stock time and price data for two stocks The return values
    (d1,p1,d2,p2) are the trade time (in days) and prices for stocks 1
    and 2 (intc and aapl)
    """
    ticker1, ticker2 = 'INTC', 'AAPL'

    file1 = cbook.get_sample_data('INTC.dat.gz')
    file2 = cbook.get_sample_data('AAPL.dat.gz')
    M1 = fromstring(file1.read(), '<d')

    M1 = resize(M1, (M1.shape[0]//2, 2))

    M2 = fromstring(file2.read(), '<d')
    M2 = resize(M2, (M2.shape[0]//2, 2))

    d1, p1 = M1[:, 0], M1[:, 1]
    d2, p2 = M2[:, 0], M2[:, 1]
    return (d1, p1, d2, p2)


d1, p1, d2, p2 = get_two_stock_data()

fig, ax = plt.subplots()
lines1 = plt.plot(d1, p1, 'b', label="INTC")
lines2 = plt.plot(d2, p2, 'r', label="AAPL")
plt.xlabel('Days')
plt.ylabel('Normalized price')
plt.xlim(0, 3)
ax.xaxis.set_major_locator(MultipleLocator(1))

plt.title('INTC vs AAPL')
plt.legend()

plt.show()
