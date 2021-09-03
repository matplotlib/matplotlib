"""
=================
Images with units
=================
Plotting images with units.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""
import numpy as np
import matplotlib.pyplot as plt
from basic_units import secs

data = np.array([[1, 2],
                 [3, 4]]) * secs

fig, ax = plt.subplots()
image = ax.imshow(data)
fig.colorbar(image)
plt.show()
