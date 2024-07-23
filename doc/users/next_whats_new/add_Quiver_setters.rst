Add ``U``, ``V`` and ``C`` setter to ``Quiver``
-----------------------------------------------

The ``U``, ``V`` and ``C`` values of the `~matplotlib.quiver.Quiver`
can now be changed after the collection has been created.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib.quiver import Quiver
    import numpy as np

    fig, ax = plt.subplots()
    X = np.arange(-10, 10, 1)
    Y = np.arange(-10, 10, 1)
    U, V = np.meshgrid(X, Y)
    C = np.hypot(U, V)
    # When X and Y are 1D and U, V are 2D, X, Y are expanded to 2D
    # using X, Y = np.meshgrid(X, Y)
    qc = ax.quiver(X, Y, U, V, C)

    qc.set_U(U/5)

    # The number of arrows can also be changed.

    # Get new X, Y, U, V, C
    X = np.arange(-10, 10, 2)
    Y = np.arange(-10, 10, 2)
    U, V = np.meshgrid(X, Y)
    C = np.hypot(U, V)
    # Use 2D X, Y coordinate (X, Y will not be expanded to 2D)
    X, Y = np.meshgrid(X, Y)

    # Set new values
    qc.set_XYUVC(X, Y, U, V, C)
