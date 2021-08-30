API Changes in 1.5.3
====================

``ax.plot(..., marker=None)`` gives default marker
--------------------------------------------------

Prior to 1.5.3 keyword arguments passed to `~matplotlib.axes.Axes.plot` were
handled in two parts -- default keyword arguments generated internal to
`~matplotlib.axes.Axes.plot` (such as the cycled styles) and user supplied
keyword arguments.  The internally generated keyword arguments were passed to
the `matplotlib.lines.Line2D` and the user keyword arguments were passed to
``ln.set(**kwargs)`` to update the artist after it was created.  Now both sets
of keyword arguments are merged and passed to `~matplotlib.lines.Line2D`.  This
change was made to allow *None* to be passed in via the user keyword arguments
to mean 'do the default thing' as is the convention through out Matplotlib
rather than raising an exception.

Unlike most `~matplotlib.lines.Line2D` setter methods
`~matplotlib.lines.Line2D.set_marker` did accept `None` as a valid
input which was mapped to 'no marker'.  Thus, by routing this
``marker=None`` through ``__init__`` rather than ``set(...)`` the meaning
of ``ax.plot(..., marker=None)`` changed from 'no markers' to 'default markers
from rcparams'.

This is change is only evident if ``mpl.rcParams['lines.marker']`` has a value
other than ``'None'`` (which is string ``'None'`` which means 'no marker').
