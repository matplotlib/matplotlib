Custom scales no longer silently discard data when their
``limit_range_for_scale`` implementation fails
-------------------------------------------------

`~.ScaleBase.val_in_range` previously caught ``TypeError`` and ``ValueError``
raised by a custom scale's `~.ScaleBase.limit_range_for_scale` and returned an
all-``False`` mask, which caused the data to be silently blanked on the plot
(e.g. in 3D).  Such errors are now propagated to the user instead, making
misbehaving custom scales much easier to debug.
