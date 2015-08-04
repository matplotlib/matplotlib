'''
Demonstration of quiver and quiverkey functions. This is using the
new version coming from the code in quiver.py.

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

# 1
plt.figure()
Q = plt.quiver(U, V)
qk = plt.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
                   fontproperties={'weight': 'bold'})
l, r, b, t = plt.axis()
dx, dy = r - l, t - b
plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])

plt.title('Minimal arguments, no kwargs')

# 2
plt.figure()
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.95, 2, r'$2 \frac{m}{s}$',
                   labelpos='E',
                   coordinates='figure',
                   fontproperties={'weight': 'bold'})
plt.axis([-1, 7, -1, 7])
plt.title('scales with plot width, not view')

# 3
plt.figure()
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', color='r', units='inches')
qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
                   fontproperties={'weight': 'bold'})
plt.plot(X[::3, ::3], Y[::3, ::3], 'k.')
plt.axis([-1, 7, -1, 7])
plt.title("pivot='mid'; every third arrow; units='inches'")

# 4
plt.figure()
M = np.hypot(U, V)
Q = plt.quiver(X, Y, U, V, M,
               units='x',
               pivot='tip',
               width=0.022,
               scale=1 / 0.15)
qk = plt.quiverkey(Q, 0.9, 1.05, 1, r'$1 \frac{m}{s}$',
                   labelpos='E',
                   fontproperties={'weight': 'bold'})
plt.plot(X, Y, 'k.')
plt.axis([-1, 7, -1, 7])
plt.title("scales with x view; pivot='tip'")

# 5
plt.figure()
Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               color='r', units='x',
               linewidths=(2,), edgecolors=('k'), headaxislength=5)
qk = plt.quiverkey(Q, 0.5, 0.03, 1, r'$1 \frac{m}{s}$',
                   fontproperties={'weight': 'bold'})
plt.axis([-1, 7, -1, 7])
plt.title("triangular head; scale with x view; black edges")

# 6
plt.figure()
M = np.zeros(U.shape, dtype='bool')
M[U.shape[0]/3:2*U.shape[0]/3,
  U.shape[1]/3:2*U.shape[1]/3] = True
U = ma.masked_array(U, mask=M)
V = ma.masked_array(V, mask=M)
Q = plt.quiver(U, V)
qk = plt.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
                   fontproperties={'weight': 'bold'})
l, r, b, t = plt.axis()
dx, dy = r - l, t - b
plt.axis([l - 0.05 * dx, r + 0.05 * dx, b - 0.05 * dy, t + 0.05 * dy])
plt.title('Minimal arguments, no kwargs - masked values')


plt.show()
