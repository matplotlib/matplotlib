"""
===============
Subplot Toolbar
===============

Matplotlib has a toolbar available for adjusting subplot spacing.
"""
import matplotlib.pyplot as plt
import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(np.random.random((100, 100)))

axs[0, 1].imshow(np.random.random((100, 100)))

axs[1, 0].imshow(np.random.random((100, 100)))

axs[1, 1].imshow(np.random.random((100, 100)))

plt.subplot_tool()
plt.show()
