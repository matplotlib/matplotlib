Improved selection of log-scale ticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm for selecting log-scale ticks (on powers of ten) has been
improved.  In particular, it will now always draw as many ticks as possible
(e.g., it will not draw a single tick if it was possible to fit two ticks); if
subsampling ticks, it will prefer putting ticks on integer multiples of the
subsampling stride (e.g., it prefers putting ticks at 10\ :sup:`0`, 10\ :sup:`3`,
10\ :sup:`6` rather than 10\ :sup:`1`, 10\ :sup:`4`, 10\ :sup:`7`) if this
results in the same number of ticks at the end; and it is now more robust
against floating-point calculation errors.
