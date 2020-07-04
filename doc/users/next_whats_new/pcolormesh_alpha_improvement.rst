pcolormesh has improved transparency handling by enabling snapping
------------------------------------------------------------------

Due to how the snapping keyword argument was getting passed to the AGG backend,
previous versions of Matplotlib would appear to show lines between the grid
edges of a mesh with transparency. This version now applies snapping
by default. To restore the old behavior (e.g., for test images), you may set
:rc:`pcolormesh.snap` to `False`.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   # Use old pcolormesh snapping values
   plt.rcParams['pcolormesh.snap'] = False
   fig, ax = plt.subplots()
   xx, yy = np.meshgrid(np.arange(10), np.arange(10))
   z = (xx + 1) * (yy + 1)
   mesh = ax.pcolormesh(xx, yy, z, shading='auto', alpha=0.5)
   fig.colorbar(mesh, orientation='vertical')
   ax.set_title('Before (pcolormesh.snap = False)')

Note that there are lines between the grid boundaries of the main plot which
are not the same transparency. The colorbar also shows these lines when a
transparency is added to the colormap because internally it uses pcolormesh
to draw the colorbar. With snapping on by default (below), the lines
at the grid boundaries disappear.

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   fig, ax = plt.subplots()
   xx, yy = np.meshgrid(np.arange(10), np.arange(10))
   z = (xx + 1) * (yy + 1)
   mesh = ax.pcolormesh(xx, yy, z, shading='auto', alpha=0.5)
   fig.colorbar(mesh, orientation='vertical')
   ax.set_title('After (default: pcolormesh.snap = True)')
