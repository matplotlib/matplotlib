``MultiCursor`` now supports Axes split over multiple figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, `.MultiCursor` only worked if all target Axes belonged to the same
figure.

As a consequence of this change, the first argument to the `.MultiCursor`
constructor has become unused (it was previously the joint canvas of all Axes,
but the canvases are now directly inferred from the list of Axes).
