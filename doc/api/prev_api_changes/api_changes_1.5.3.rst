Changes in 1.5.3
================

``ax.plot(..., marker=None)`` gives default marker
--------------------------------------------------

Prior to 1.5.3 kwargs passed to `~matplotlib.Axes.plot` were handled
in two parts -- default kwargs generated internal to
`~matplotlib.Axes.plot` (such as the cycled styles) and user supplied
kwargs.  The internally generated kwargs were passed to the
`matplotlib.lines.Line2D.__init__` and the user kwargs were passed to
``ln.set(**kwargs)`` to update the artist after it was created.  Now
both sets of kwargs are merged and passed to
`~matplotlib.lines.Line2D.__init__`.  This change was made to allow `None`
to be passed in via the user kwargs to mean 'do the default thing'  as
is the convention through out mpl rather than raising an exception.

Unlike most `~matplotlib.lines.Line2D` setter methods
`~matplotlib.lines.Line2D.set_marker` did accept `None` as a valid
input which was mapped to 'no marker'.  Thus, by routing this
``marker=None`` through ``__init__`` rather than ``set(...)`` the meaning
of ``ax.plot(..., marker=None)`` changed from 'no markers' to 'default markers
from rcparams'.

This is change is only evident if ``mpl.rcParams['lines.marker']`` has a value
other than ``'None'`` (which is string ``'None'`` which means 'no marker').
