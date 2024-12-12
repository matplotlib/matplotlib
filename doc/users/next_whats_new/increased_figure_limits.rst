Increased Figure limits with Agg renderer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Figures using the Agg renderer are now limited to 2**23 pixels in each
direction, instead of 2**16. Additionally, bugs that caused artists to not
render past 2**15 pixels horizontally have been fixed.

Note that if you are using a GUI backend, it may have its own smaller limits
(which may themselves depend on screen size.)
