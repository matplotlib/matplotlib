``PatchCollection`` legends now supported
------------------------------------------
`.PatchCollection` instances now properly display in legends when given a label.
Previously, labels on `~.PatchCollection` objects were ignored by the legend
system, requiring users to create manual legend entries.

The legend entry displays a rectangle matching the visual properties (colors,
line styles, line widths) of the first patch in the collection.

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots()
    patches = [mpatches.Circle((0, 0), 0.1), mpatches.Rectangle((0.5, 0.5), 0.2, 0.3)]
    pc = PatchCollection(patches, facecolor='blue', edgecolor='black', label='My patches')
    ax.add_collection(pc)
    ax.legend()  # Now displays the label "My patches"
    plt.show()

This fix resolves :ghissue:`23998`.
