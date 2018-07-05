`.FuncAnimation` now draws artists according to their zorder when blitting
--------------------------------------------------------------------------

`.FuncAnimation` now draws artists returned by the user-
function according to their zorder when using blitting,
instead of using the order in which they are being passed.
However, note that only zorder of passed artists will be
respected, as they are drawn on top of any existing artists
(see `#11369 <https://github.com/matplotlib/matplotlib/issues/11369>`_).