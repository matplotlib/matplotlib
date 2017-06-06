Axis-sharing between figures is blocked
```````````````````````````````````````

An attempt to share an axis between ``Axes`` objects in different
figures will now raise a ValueError.  Previously such sharing
was allowed but could lead to an endless loop when combined
with fixed aspect ratios, as in the case of ``imshow``, for
example.
