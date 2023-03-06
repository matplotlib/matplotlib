import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

"""""
Should create the same plots, running the saved figures in an
external png difference checker gives the same result after the
new implementation.
"""

x = np.arange(0, 10)
norm = plt.Normalize(0, 10)
cmap = mpl.colormaps['viridis'].resampled(10)
cols = cmap(norm(x))

#Testcase 1:
#plt.scatter(x, x, c=x, facecolors='none') 

#Testcase 2:
plt.scatter(x, x, facecolors='none', edgecolors=cols)
plt.show()