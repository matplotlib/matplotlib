``pcolormesh`` accepts RGB(A) colors
------------------------------------

The `~.Axes.pcolormesh` method can now handle explicit colors
specified with RGB(A) values. To specify colors, the array must be 3D
with a shape of ``(M, N, [3, 4])``.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np

    colors = np.linspace(0, 1, 90).reshape((5, 6, 3))
    plt.pcolormesh(colors)
    plt.show()
