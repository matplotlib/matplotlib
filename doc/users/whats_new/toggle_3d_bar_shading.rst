Users can now toggle shading in 3D bar plots
--------------------------------------------

A new ``shade`` parameter has been added the 3D bar plotting method.
The default behavior remains to shade the bars, but now users
have the option of setting ``shade`` to ``False``.


Example
```````
::

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(7,3))
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.arange(2)
    y = np.arange(3)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d
    ax1.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()
