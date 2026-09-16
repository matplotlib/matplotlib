Pie charts are now a single artist
----------------------------------

`.Axes.pie` now returns a `.Pie` artist instead of a ``PieContainer`` (which
is deprecated).  The `.Pie` collects the wedge patches, the optional shadow
patches and all labels as child artists, so the whole chart can be treated
as one object::

    pie = ax.pie([1, 2, 3], shadow=True)
    pie.remove()

Because the wedges and labels are children of the `.Pie`, they are no longer
added to `.Axes.patches` and `.Axes.texts` directly.  Access them through
`~.Pie.wedges` and `~.Pie.texts` instead.

Compound artists can provide their own legend entries by implementing
`~.Pie.get_legend_handles`; `.Axes.legend` uses this to show the individual
wedges of a pie.
