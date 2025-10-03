``violin_stats`` simpler *method* parameter
-------------------------------------------

The *method* parameter of `~.cbook.violin_stats` may now be specified as tuple of
strings, and has a new default ``("GaussianKDE", "scott")``.  Calling
`~.cbook.violin_stats` followed by `~.Axes.violin` is therefore now equivalent to
calling `~.Axes.violinplot`.

.. plot::
    :include-source: true
    :alt: Example showing violin_stats followed by violin gives the same result as violinplot

    import matplotlib.pyplot as plt
    from matplotlib.cbook import violin_stats
    import numpy as np

    rng = np.random.default_rng(19680801)
    data = rng.normal(size=(10, 3))

    fig, (ax1, ax2) = plt.subplots(ncols=2, layout='constrained', figsize=(6.4, 3.5))

    # Create the violin plot in one step
    ax1.violinplot(data)
    ax1.set_title('One Step')

    # Process the data and then create the violin plot
    vstats = violin_stats(data)
    ax2.violin(vstats)
    ax2.set_title('Two Steps')

    plt.show()
