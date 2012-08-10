import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi, num=40, endpoint=True)
y = np.sin(x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.stairs(x, y)
plt.show()
