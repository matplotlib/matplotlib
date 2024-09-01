Separated ``hatchcolor`` from ``edgecolor``
-------------------------------------------

The ``hatchcolor`` parameter for `~matplotlib.patches.Patch` objects has been
separated from the ``edgecolor`` parameter. This allows for using different colors
for hatches and edges by explicitly setting the ``hatchcolor`` parameter.

.. plot::
    :include-source: true
    :alt: Example of using the hatchcolor parameter in a Rectangle

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()
    patch = Rectangle((0.2, 0.2), 0.6, 0.6, edgecolor='red', linewidth=2,
                      hatch='//', hatchcolor='orange')
    ax.add_patch(patch)
    plt.show()
