``edgegapcolor`` for Patches
----------------------------

`~matplotlib.patches.Patch` now supports an *edgegapcolor* parameter,
similar to the existing *gapcolor* in `.Line2D`. This allows patches with
dashed edges to display a secondary color in the gaps, creating a "striped"
edge effect.

This is useful when drawing unfilled patches on backgrounds of unknown color,
where alternating edge colors ensure the patch boundary remains visible.

.. plot::
    :include-source: true
    :alt: A rectangle with a dashed orange edge and blue gaps, demonstrating the edgegapcolor feature.

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()
    rect = Rectangle((0.1, 0.1), 0.6, 0.6, fill=False,
                      edgecolor='orange', edgegapcolor='blue',
                      linestyle='--', linewidth=3)
    ax.add_patch(rect)
    plt.show()
