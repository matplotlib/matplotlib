``pcolormesh`` and ``pcolor`` now respond correctly to ``relim``
----------------------------------------------------------------

`~.Axes.pcolormesh` and `~.Axes.pcolor` now correctly participate in
`~.Axes.relim` / `~.Axes.autoscale_view`. Previously, calling
``ax.relim(); ax.autoscale_view()`` after adding a mesh with ``pcolormesh``
or ``pcolor`` would reset the axis limits to the default empty-axes range
instead of the data range of the mesh::

    fig, ax = plt.subplots()
    ax.pcolormesh([[0, 1], [2, 3]], [[0, 0], [1, 1]], [[0.5]])
    ax.relim()
    ax.autoscale_view()
    # Previously gave (-0.055, 0.055); now correctly gives (0, 2)
    print(ax.get_xlim())
