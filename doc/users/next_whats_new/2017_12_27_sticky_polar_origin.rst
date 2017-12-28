Autoscaling a polar plot snaps to the origin
--------------------------------------------

Setting the limits automatically in a polar plot now snaps the radial limit
to zero if the automatic limit is nearby. This means plotting from zero doesn't
automatically scale to include small negative values on the radial axis.

The limits can still be set manually in the usual way using `set_ylim`.
