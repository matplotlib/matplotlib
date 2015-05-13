Changed snap threshold for circle markers to inf
````````````````````````````````````````````````

When drawing circle markers above some marker size (previously 6.0)
the path used to generate the marker was snapped to pixel centers.  However,
this ends up distorting the marker away from a circle.  By setting the
snap threshold to inf snapping is never done on circles.

This change broke several tests, but is an improvement.
