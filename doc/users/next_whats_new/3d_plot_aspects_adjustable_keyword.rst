*adjustable* keyword argument for setting equal aspect ratios in 3D
-------------------------------------------------------------------

While setting equal aspect ratios for 3D plots, users can choose to modify
either the data limits or the bounding box.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations, product

    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz')
    fig, axs = plt.subplots(1, len(aspects), subplot_kw={'projection': '3d'},
                            figsize=(12, 6))

    # Draw rectangular cuboid with side lengths [4, 3, 5]
    r = [0, 1]
    scale = np.array([4, 3, 5])
    pts = combinations(np.array(list(product(r, r, r))), 2)
    for start, end in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            for ax in axs:
                ax.plot3D(*zip(start*scale, end*scale), color='C0')

    # Set the aspect ratios
    for i, ax in enumerate(axs):
        ax.set_aspect(aspects[i], adjustable='datalim')
        # Alternatively: ax.set_aspect(aspects[i], adjustable='box')
        # which will change the box aspect ratio instead of axis data limits.
        ax.set_title(f"set_aspect('{aspects[i]}')")

    plt.show()
