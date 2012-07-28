import numpy as np
from matplotlib import pyplot as plt

fnx = lambda : np.random.randint(5, 50, 10)
y = np.row_stack((fnx(), fnx(), fnx()))
x = np.arange(10)

y1, y2, y3 = fnx(), fnx(), fnx()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.stackplot(x, y)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.stackplot(x, y1, y2, y3)
plt.show()
