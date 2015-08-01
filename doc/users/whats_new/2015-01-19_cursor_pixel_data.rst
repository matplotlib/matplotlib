Allow Artists to Display Pixel Data in Cursor
---------------------------------------------

Adds `get_cursor_data` and `format_cursor_data` methods to artists
which can be used to add zdata to the cursor display
in the status bar.  Also adds an implementation for Images.

Added an property attribute ``mouseover`` and rcParam
(``axes.mouseover``) to Axes objects to control if the hit list is
computed.  For moderate number of artists (>100) in the axes the
expense to compute the top artist becomes greater than the time
between mouse events.  For this reason the behavior defaults to
``False``, but maybe enabled by default in the future when the hitlist
computation is optimized.

To enable the cursor message on a given axes ::

  ax.mouseover = True

To enable for all new axes created ::

  matplotlib.rcParams['axes.mouseover'] = True
