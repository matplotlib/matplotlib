3D annotations can anchor to 3D coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.annotate` now accepts 3D data coordinates:
when ``xycoords='data'``, the annotated position *xy* may be passed as a
3-tuple ``(x, y, z)``.  In that case, the annotation is projected during
draws, so it stays attached to the intended point when rotating or zooming the
3D view.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   fig = plt.figure()
   ax = fig.add_subplot(projection="3d")

   x = y = z = np.arange(10)
   ax.scatter(x, y, z, s=20)

   ax.annotate(
       "3D anchor",
       (x[5], y[5], z[5]),
       xytext=(10, 10),
       textcoords="offset points",
       arrowprops=dict(arrowstyle="->"),
   )

   plt.show()
