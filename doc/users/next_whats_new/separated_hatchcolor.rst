Separated ``hatchcolor`` from ``edgecolor``
-------------------------------------------

When the *hatchcolor* parameter is specified, it will be used for the hatch.
If it is not specified, it will fall back to using :rc:`hatch.color`.
The special value 'edge' uses the patch edgecolor, with a fallback to
:rc:`patch.edgecolor` if the patch edgecolor is 'none'.
Previously, hatch colors were the same as edge colors, with a fallback to
:rc:`hatch.color` if the patch did not have an edge color.

.. plot::
    :include-source: true
    :alt: Four Rectangle patches, each displaying the color of hatches in different specifications of edgecolor and hatchcolor. Top left has hatchcolor='black' representing the default value when both hatchcolor and edgecolor are not set, top right has edgecolor='blue' and hatchcolor='black' which remains when the edgecolor is set again, bottom left has edgecolor='red' and hatchcolor='orange' on explicit specification and bottom right has edgecolor='green' and hatchcolor='green' when the hatchcolor is not set.

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    # In this case, hatchcolor is orange
    patch1 = Rectangle((0.1, 0.1), 0.3, 0.3, edgecolor='red', linewidth=2,
                       hatch='//', hatchcolor='orange')
    ax.add_patch(patch1)

    # When hatchcolor is not specified, it matches edgecolor
    # In this case, hatchcolor is green
    patch2 = Rectangle((0.6, 0.1), 0.3, 0.3, edgecolor='green', linewidth=2,
                       hatch='//', facecolor='none')
    ax.add_patch(patch2)

    # If both hatchcolor and edgecolor are not specified
    # it will default to the 'patch.edgecolor' rcParam, which is black by default
    # In this case, hatchcolor is black
    patch3 = Rectangle((0.1, 0.6), 0.3, 0.3, hatch='//')
    ax.add_patch(patch3)

    # When using `hatch.color` in the `rcParams`
    # edgecolor will now not overwrite hatchcolor
    # In this case, hatchcolor is black
    with plt.rc_context({'hatch.color': 'black'}):
        patch4 = Rectangle((0.6, 0.6), 0.3, 0.3, edgecolor='blue', linewidth=2,
                           hatch='//', facecolor='none')

    # hatchcolor is black (it uses the `hatch.color` rcParam value)
    patch4.set_edgecolor('blue')
    # hatchcolor is still black (here, it does not update when edgecolor changes)
    ax.add_patch(patch4)

    ax.annotate("hatchcolor = 'orange'",
                xy=(.5, 1.03), xycoords=patch1, ha='center', va='bottom')
    ax.annotate("hatch color unspecified\nedgecolor='green'",
                xy=(.5, 1.03), xycoords=patch2, ha='center', va='bottom')
    ax.annotate("hatch color unspecified\nusing patch.edgecolor",
                xy=(.5, 1.03), xycoords=patch3, ha='center', va='bottom')
    ax.annotate("hatch.color='black'",
                xy=(.5, 1.03), xycoords=patch4, ha='center', va='bottom')

    plt.show()
