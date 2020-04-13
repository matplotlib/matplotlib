`.Axes.sharex`, `.Axes.sharey`
------------------------------
These new methods allow sharing axes *immediately* after creating them.  For
example, they can be used to selectively link some axes created all together
using `~.Figure.subplots`.

Note that they may *not* be used to share axes after any operation (e.g.,
drawing) has occurred on them.
