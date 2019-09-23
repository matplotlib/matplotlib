Fair DivergingNorm
------------------
`~.DivergingNorm` now has an argument ``fair``, which can be set to ``True``
in order to create an off-centered normalization with equally spaced colors.

..plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import DivergingNorm

    np.random.seed(19680801)
    data = np.random.rand(4, 11)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 2))

    norm1 = DivergingNorm(0.25, vmin=0, vmax=1, fair=False)
    im = ax1.imshow(data, cmap='RdBu', norm=norm1)
    cbar = fig.colorbar(im, ax=ax1, orientation="horizontal", aspect=15)

    norm2 = DivergingNorm(0.25, vmin=0, vmax=1, fair=True)
    im = ax2.imshow(data, cmap='RdBu', norm=norm2)
    cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", aspect=15)

    ax1.set_title("DivergingNorm(.., fair=False)")
    ax2.set_title("DivergingNorm(.., fair=True)")
    plt.show()