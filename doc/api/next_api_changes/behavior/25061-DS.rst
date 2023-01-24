TwoSlopeNorm now auto-expands to always have two slopes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the case where either ``vmin`` or ``vmax`` are not manually specified
to `~.TwoSlopeNorm`, and where the data it is scaling is all less than or
greater than the center point, the limits are now auto-expanded so there
are two symmetrically sized slopes either side of the center point.

Previously ``vmin`` and ``vmax`` were clipped at the center point, which
caused issues when displaying color bars.

This does not affect behaviour when ``vmin`` and ``vmax`` are manually
specified by the user.
