"""
=============================
Whats New 0.98.4 Fill Between
=============================
 generates an example of using the fill_between() method with opposing two quadratics
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.01)
y1 = -5*x*x + x + 10
y2 = 5*x*x + x

fig, ax = plt.subplots()
ax.plot(x, y1, x, y2, color='black')
ax.fill_between(x, y1, y2, where=y2>y1, facecolor='yellow',alpha=0.5)
ax.fill_between(x, y1, y2, where=y2<=y1, facecolor='red',alpha=0.5)
ax.set_title('Highlight Between and Intercept')

plt.show()
