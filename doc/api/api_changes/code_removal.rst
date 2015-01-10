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


Remove ``set_colorbar`` method from ``ScalarMappable``
------------------------------------------------------
Remove ``set_colorbar`` method, use `colorbar` attribute directly.


patheffects.svg
---------------
 - remove ``get_proxy_renderer`` method from ``AbstarctPathEffect`` class
 - remove ``patch_alpha`` and ``offset_xy`` from ``SimplePatchShadow``


Remove ``testing.image_util.py``
--------------------------------
Contained only a no-longer used port of functionality from PIL


Remove ``mlab.FIFOBuffer``
--------------------------
Not used internally and not part of core mission of mpl.
