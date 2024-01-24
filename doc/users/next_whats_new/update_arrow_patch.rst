Update the position of arrow patch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adds a setter method that allows the user to update the position of the
`.patches.Arrow` object without requiring a full re-draw.

.. plot::
    :include-source: true
    :alt: Example of changing the position of the arrow with the new ``set_data`` method.

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arrow
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    a = mpl.patches.Arrow(2, 0, 0, 10)
    ax.add_patch(a)


    # code for modifying the arrow
    def update(i):
        a.set_data(x=.5, dx=i, dy=6, width=2)


    ani = animation.FuncAnimation(fig, update, frames=15, interval=90, blit=False)

    plt.show()
