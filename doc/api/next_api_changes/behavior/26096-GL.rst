``ScalarMappable.to_rgba()`` now respects the mask of RGB(A) arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, the mask was ignored. Now the alpha channel is set to 0 if any
component (R, G, B, or A) is masked.
