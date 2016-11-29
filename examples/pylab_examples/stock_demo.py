import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator
from data_helper import get_two_stock_data

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
