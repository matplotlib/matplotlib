Code Removal
````````````

Legend
------
 - Removed handling of `loc` as a positional argument to `Legend`


Legend handlers
~~~~~~~~~~~~~~~
Remove code to allow legend handlers to be callable.  They must now
implement a method ``legend_artist``.


Axis
----
Removed method ``set_scale``.  This is now handled via a private method which
should not be used directly by users.  It is called via ``Axes.set_{x,y}scale``
which takes care of ensuring the coupled changes are also made to the Axes object.

finance.py
----------
Removed functions with ambiguous argument order from finance.py


Annotation
----------
Removed ``textcoords`` and ``xytext`` proprieties from Annotation objects.


spinxext.ipython_*.py
---------------------
Both ``ipython_console_highlighting`` and ``ipython_directive`` have been moved to
`IPython`.

Change your import from 'matplotlib.sphinxext.ipython_directive' to
'IPython.sphinxext.ipython_directive' and from 'matplotlib.sphinxext.ipython_directive' to
'IPython.sphinxext.ipython_directive'


LineCollection.color
--------------------
Deprecated in 2005, use ``set_color``


remove 'faceted' as a valid value for `shading` in ``tri.tripcolor``
--------------------------------------------------------------------
Use `edgecolor` instead.  Added validation on ``shading`` to
only be valid values.
