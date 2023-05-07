Specify ticks and axis label positions for 3D plots
---------------------------------------------------

You can now specify the positions of ticks and axis labels for 3D plots.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt

   positions = ['default', 'upper', 'lower', 'both', 'none']
   fig, axs = plt.subplots(1, 5, subplot_kw={'projection': '3d'})
   for ax, pos in zip(axs, positions):
       for axis in ax.xaxis, ax.yaxis, ax.zaxis:
           axis._label_position = pos
           axis._tick_position = pos
       ax.set(xlabel='x', ylabel='y', zlabel='z', title=pos)
