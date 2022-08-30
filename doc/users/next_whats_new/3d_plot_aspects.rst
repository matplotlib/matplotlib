Set equal aspect ratio for 3D plots
-----------------------------------

Users can set the aspect ratio for the X, Y, Z axes of a 3D plot to be 'equal',
'equalxy', 'equalxz', or 'equalyz' rather than the default of 'auto'.

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import combinations, product

    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz')
    fig, axs = plt.subplots(1, len(aspects), subplot_kw={'projection': '3d'})

    # Draw rectangular cuboid with side lengths [1, 1, 5]
    r = [0, 1]
    scale = np.array([1, 1, 5])
    pts = combinations(np.array(list(product(r, r, r))), 2)
    for start, end in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            for ax in axs:
                ax.plot3D(*zip(start*scale, end*scale), color='C0')

    # Set the aspect ratios
    for i, ax in enumerate(axs):
        ax.set_box_aspect((3, 4, 5))
        ax.set_aspect(aspects[i])
        ax.set_title(f"set_aspect('{aspects[i]}')")

    fig.set_size_inches(13, 3)
    plt.show()
