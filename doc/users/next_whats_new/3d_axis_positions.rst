Specify ticks and axis label positions for 3D plots
---------------------------------------------------

You can now specify the positions of ticks and axis labels for 3D plots.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt

   positions = ['lower', 'upper', 'default', 'both', 'none']
   fig, axs = plt.subplots(2, 3, figsize=(12, 8),
                           subplot_kw={'projection': '3d'})
   for ax, pos in zip(axs.flatten(), positions):
       for axis in ax.xaxis, ax.yaxis, ax.zaxis:
           axis.set_label_position(pos)
           axis.set_ticks_position(pos)
       title = f'position="{pos}"'
       ax.set(xlabel='x', ylabel='y', zlabel='z', title=title)
   axs[1, 2].axis('off')
