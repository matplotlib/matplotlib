Add a new valid color format ``(matplotlib_color, alpha)``
----------------------------------------------------------


.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    rectangle = Rectangle((.2, .2), .6, .6,
                          facecolor=('blue', 0.2),
                          edgecolor=('green', 0.5))
    ax.add_patch(rectangle)


Users can define a color using the new color specification, *(matplotlib_color, alpha)*.
Note that an explicit alpha keyword argument will override an alpha value from
*(matplotlib_color, alpha)*.
