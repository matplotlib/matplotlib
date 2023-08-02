Add ``RectangleCollection``
---------------------------

The `~matplotlib.collections.RectangleCollection` is added to create collection of `~matplotlib.patches.Rectangle`

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib.collections import RectangleCollection
    import numpy as np

    rng = np.random.default_rng(0)

    widths = (2, )
    heights = (3, )
    angles = (45, )
    offsets = rng.random((10, 2)) * 10

    fig, ax = plt.subplots()

    ec = RectangleCollection(
        widths=widths,
        heights=heights,
        angles=angles,
        offsets=offsets,
        units='x',
        offset_transform=ax.transData,
        )

    ax.add_collection(ec)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
