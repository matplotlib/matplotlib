"""
=========================
triplot(x, y, [triangle])
=========================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('cheatsheet_gallery')

# make structured data
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))

# sample it to make x, y, z
np.random.seed(1)
ysamp = np.random.randint(0, high=256, size=250)
xsamp = np.random.randint(0, high=256, size=250)
y = Y[:, 0][ysamp]
x = X[0, :][xsamp]

# plot:
fig, ax = plt.subplots()

ax.triplot(x, y)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.show()
