New and improved 3D bar plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We fixed a long standing issue with incorrect z-sorting in 3d bar graphs.
It is now possible to produce 3D bar charts that render correctly for all
viewing angles by using `.Axes3D.bar3d_grid`. In addition, bar charts with
hexagonal cross section can now be created with `.Axes3Dx.hexbar3d`. This
supports visualisation of density maps on hexagonal tessellations of the data
space. Two new artist collections are introduced to support this functionality:
`.Bar3DCollection` and `.HexBar3DCollection`.


.. plot::
    :include-source: true
    :alt: Example of creating hexagonal 3D bars

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    bars3d = ax1.bar3d_grid([0, 1], [0, 1], [1, 2], '0.8', facecolors=('m', 'y'))
    hexbars3d = ax2.hexbar3d([0, 1], [0, 1], [1, 2], '0.8', facecolors=('m', 'y'))
    plt.show()
