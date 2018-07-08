Contour color autoscaling improvements
--------------------------------------

Selection of contour levels is now the same for contour and
contourf; previously, for contour, levels outside the data range were
deleted.  (Exception: if no contour levels are found within the
data range, the `levels` attribute is replaced with a list holding
only the minimum of the data range.)

When contour is called with levels specified as a target number rather
than a list, and the 'extend' kwarg is used, the levels are now chosen
such that some data typically will fall in the extended range.

When contour is called with a `LogNorm` or a `LogLocator`, it will now
select colors using the geometric mean rather than the arithmetic mean
of the contour levels.
