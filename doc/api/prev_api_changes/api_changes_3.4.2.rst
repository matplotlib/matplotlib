API Changes for 3.4.2
=====================

Behaviour changes
-----------------

Rename first argument to ``subplot_mosaic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both `.FigureBase.subplot_mosaic`, and `.pyplot.subplot_mosaic` have had the
first position argument renamed from *layout* to *mosaic*.  This is because we
are considering to consolidate *constrained_layout* and *tight_layout* keyword
arguments in the Figure creation functions of `.pyplot` into a single *layout*
keyword argument which would collide.

As this API is provisional, we are changing this with no deprecation period.
