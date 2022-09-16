``Poly3DCollection`` supports shading
-------------------------------------

It is now possible to shade a `.Poly3DCollection`. This is useful if the
polygons are obtained from e.g. a 3D model.

.. plot::
    :include-source: true

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Define 3D shape
    block = np.array([
        [[1, 1, 0],
         [1, 0, 0],
         [0, 1, 0]],
        [[1, 1, 0],
         [1, 1, 1],
         [1, 0, 0]],
        [[1, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[1, 0, 0],
         [1, 1, 1],
         [0, 1, 0]]
    ])

    ax = plt.subplot(projection='3d')
    pc = Poly3DCollection(block, facecolors='b', shade=True)
    ax.add_collection(pc)
    plt.show()
