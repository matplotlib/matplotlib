`Collection` offsets are no longer implicitly flattened
-------------------------------------------------------

`Collection` (and thus `scatter` -- both 2D and 3D) no longer implicitly
flattens its offsets.  As a consequence, `scatter`'s x and y arguments can no
longer be 2+-dimensional arrays.
