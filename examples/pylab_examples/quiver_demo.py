'''
Demonstration of advanced quiver and quiverkey functions.

Known problem: the plot autoscaling does not take into account
the arrows, so those on the boundaries are often out of the picture.
This is *not* an easy problem to solve in a perfectly general way.
The workaround is to manually expand the axes.
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U = np.cos(X)
V = np.sin(Y)

plt.figure()
plt.title('scales with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.95, 2, r'$2 \frac{m}{s}$',
                   labelpos='E',
                   coordinates='figure',
                   fontproperties={'weight': 'bold'})
plt.axis([-1, 7, -1, 7])

plt.figure()
plt.title("pivot='mid'; every third arrow; units='inches'")
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', color='r', units='inches')
qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
                   fontproperties={'weight': 'bold'})
plt.plot(X[::3, ::3], Y[::3, ::3], 'k.')
plt.axis([-1, 7, -1, 7])

plt.figure()
plt.title("scales with x view; pivot='tip'")
M = np.hypot(U, V)
Q = plt.quiver(X, Y, U, V, M,
               units='x',
               pivot='tip',
               width=0.022,
               scale=1 / 0.15)
qk = plt.quiverkey(Q, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                   labelpos='E',
                   fontproperties={'weight': 'bold'})
plt.plot(X, Y, 'k.', markersize=2)
plt.axis([-1, 7, -1, 7])

plt.figure()
plt.title("triangular head; scale with x view; black edges")

Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               color='r', units='x',
               linewidths=(0.5,), edgecolors=('k'), headaxislength=5)
qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
                   fontproperties={'weight': 'bold'})
plt.axis([-1, 7, -1, 7])

plt.show()
