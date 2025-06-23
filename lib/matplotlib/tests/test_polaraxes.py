import numpy as np
import matplotlib.pyplot as plt

D = np.array([[2,.3],[.3,2.5]])

# creating an ellipse)
lam, vec = np.linalg.eigh(D)
theta    = np.linspace(-np.pi,np.pi,1001)
xy       = np.stack((lam[0]*np.cos(theta), lam[1]*np.sin(theta)), axis=0)
xy       = vec.T@xy
radii    = np.sqrt(np.sum(xy**2, axis=0))
thetas   = np.arctan2(-xy[1],xy[0])

plt.figure()
# with ticks
ax1 = plt.subplot(121,projection='polar')
ax1.plot(thetas, radii)
# without ticks
ax2 = plt.subplot(122, projection='polar')
ax2.plot(thetas, radii)
ax2.set_yticks([])
ax2.set_xticks([])

plt.show()