import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyplot

x = np.linspace (0,50,400)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

pyplot.plot(x, y1, lorder=2, label="sin")
pyplot.plot(x, y2, lorder=3, label="cos")
pyplot.plot(x, y3, lorder=1, label="tan")
pyplot.legend()
pyplot.ylim(-1.5, 1.5)
pyplot.show()

