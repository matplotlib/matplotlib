Code Removal
````````````

Legend
------
Removed handling of `loc` as a positional argument to `Legend`


Axis
----
Removed method ``set_scale``.  This is now handled via a private method which
should not be used directly by users.  It is called via ``Axes.set_{x,y}scale``
which takes care of ensuring the coupled changes are also made to the Axes object.
