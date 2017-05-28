"""
===============
Subplot Toolbar
===============

"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.subplot(221)
plt.imshow(np.random.random((100, 100)))
plt.subplot(222)
plt.imshow(np.random.random((100, 100)))
plt.subplot(223)
plt.imshow(np.random.random((100, 100)))
plt.subplot(224)
plt.imshow(np.random.random((100, 100)))

plt.subplot_tool()
plt.show()
