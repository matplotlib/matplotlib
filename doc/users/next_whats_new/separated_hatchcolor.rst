Separated ``hatchcolor`` from ``edgecolor``
-------------------------------------------

`~matplotlib.patches.Patch` gained a new *hatchcolor* parameter to explicitly
control hatch colors. Previously, hatch colors were the same as edge colors,
with a fallback to :rc:`hatch.color` if the patch did not have an edge color.

Inherit Logic
~~~~~~~~~~~~~
When the *hatchcolor* parameter is specified, it will be used for the hatch.
If it is not specified, it will fallback to using :rc:`hatch.color`.

The special value 'inherit' takes over the patch edgecolor, with a fallback to
:rc:`patch.edgecolor` if the patch edgecolor is 'none'.

If the patch inherits hatchcolor from edgecolor, hatchcolor will
be updated if edgecolor is changed (for example: by calling *set_edgecolor()*).
But if hatchcolor is explicitly set (for example: by calling *set_hatchcolor()*
or by using *hatch.color* rcParam), it will not be updated when edgecolor changes.

.. plot::
    :include-source: true
    :alt: Example of using the hatchcolor parameter in a Rectangle

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    # hatchcolor can now be controlled using the `hatchcolor` parameter
    # In this case, hatchcolor is orange
    patch1 = Rectangle((0.1, 0.1), 0.3, 0.3, edgecolor='red', linewidth=2,
                      hatch='//', hatchcolor='orange')
    ax.add_patch(patch1)

    # When hatchcolor is not specified, it inherits from edgecolor
    # In this case, hatchcolor is green
    patch2 = Rectangle((0.6, 0.1), 0.3, 0.3, edgecolor='green', linewidth=2,
                       hatch='//', facecolor='none')
    ax.add_patch(patch2)

    # If both hatchcolor and edgecolor are not specified, it will default to the 'patch.edgecolor' rcParam, which is black by default
    # In this case, hatchcolor is black
    patch3 = Rectangle((0.1, 0.6), 0.3, 0.3, hatch='//')
    ax.add_patch(patch3)

    # When using `hatch.color` in the `rcParams`, edgecolor will now not overwrite hatchcolor
    # In this case, hatchcolor is black
    with plt.rc_context({'hatch.color': 'black'}):
        patch4 = Rectangle((0.6, 0.6), 0.3, 0.3, edgecolor='blue', linewidth=2,
                           hatch='//', facecolor='none')

    # hatchcolor is black (it uses the `hatch.color` rcParam value)
    assert patch4._hatch_color == mpl.colors.to_rgba('black')
    patch4.set_edgecolor('blue')
    # hatchcolor is still black (here, it does not update when edgecolor changes)
    assert patch4._hatch_color == mpl.colors.to_rgba('black')
    ax.add_patch(patch4)

    plt.show()
