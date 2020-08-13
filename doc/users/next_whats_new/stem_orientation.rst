Added *orientation* parameter for stem plots
--------------------------------------------

By default, stem lines are vertical. They can be changed to horizontal using
the *orientation* parameter of `.Axes.stem` or `.pyplot.stem`:

.. plot::

    locs = np.linspace(0.1, 2 * np.pi, 25)
    heads = np.cos(locs)

    fig, ax = plt.subplots()
    ax.stem(locs, heads, orientation='horizontal')
