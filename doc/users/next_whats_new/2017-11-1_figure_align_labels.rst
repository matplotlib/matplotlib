xlabels and ylabels can now be automatically aligned
----------------------------------------------------

Subplot axes ``ylabels`` can be misaligned horizontally if the tick labels
are very different widths.  The same can happen to ``xlabels`` if the
ticklabels are rotated on one subplot (for instance).  The new methods
on the `Figure` class: `Figure.align_xlabels` and `Figure.align_ylabels`
will now align these labels horizontally or vertically.  If the user only
wants to align some axes, a list of axes can be passed.  If no list is
passed, the algorithm looks at all the labels on the figure.

Only labels that have the same subplot locations are aligned.  i.e. the
ylabels are aligned only if the subplots are in the same column of the
subplot layout.

Alignemnt is persistent and automatic after these are called.

A convenience wrapper `Figure.align_labels` calls both functions at once.

.. plot::

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(5, 3), tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax = fig.add_subplot(gs[0,:])
    ax.plot(np.arange(0, 1e6, 1000))
    ax.set_ylabel('Test')
    for i in range(2):
        ax = fig.add_subplot(gs[1, i])
        ax.set_ylabel('Booooo')
        ax.set_xlabel('Hello')
        if i == 0:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
    fig.align_labels()
