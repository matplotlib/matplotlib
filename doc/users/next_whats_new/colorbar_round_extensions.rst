Round extensions in colorbars
-----------------------------

Colorbars can now have round extensions using the keyword
extend_shape, while the radius of the corners can be controlled with
the keyword rounding_size (min=0, max=0.5).

.. plot::
    :include-source: true

    import numpy as np
    import matplotlib.pyplot as plt
    N = 37
    x, y = np.mgrid[:N, :N]
    Z = (np.cos(x*0.2) + np.sin(y*0.3))
    cmap = plt.get_cmap("plasma")
    fig, ax1 = plt.subplots(figsize=(5, 4))
    pos = ax1.imshow(Z, cmap=cmap, interpolation='bicubic')
    fig.colorbar(pos, ax=ax1, extend='both', extend_shape='round', rounding_size=0.4, aspect=15)
    plt.show()

