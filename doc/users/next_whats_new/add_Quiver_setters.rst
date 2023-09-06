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
    qc = ax.quiver(X, Y, U, V, C)

    qc.set_U(U/5)


The number of arrows can also be changed.

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
    qc = ax.quiver(X, Y, U, V, C)

    # Get new X, Y, U, V, C
    X = np.arange(-10, 10, 2)
    Y = np.arange(-10, 10, 2)
    U, V = np.meshgrid(X, Y)
    C = np.hypot(U, V)
    X, Y = np.meshgrid(X, Y)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    # Set new values
    qc.set_offsets(XY)
    qc.set_UVC(U, V, C)
