"""
==============
Dropped spines
==============

Demo of spines offset from the axes (a.k.a. "dropped spines").
"""
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()

image = np.random.uniform(size=(10, 10))
ax.imshow(image, cmap=plt.cm.gray)
ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines[['left', 'bottom']].set_position(('outward', 10))
# Hide the right and top spines
ax.spines[['top', 'right']].set_visible(False)

plt.show()
