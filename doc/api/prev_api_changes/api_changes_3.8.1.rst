API Changes for 3.8.1
=====================

Behaviour
---------

Default behaviour of ``hexbin`` with *C* provided requires at least 1 point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The behaviour changed in 3.8.0 to be inclusive of *mincnt*. However, that resulted in
errors or warnings with some reduction functions, so now the default is to require at
least 1 point to call the reduction function. This effectively restores the default
behaviour to match that of Matplotlib 3.7 and before.


Deprecations
------------

Deprecations removed in ``contour``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``contour.allsegs``, ``contour.allkinds``, and ``contour.find_nearest_contour`` are no
longer marked for deprecation.


Development
-----------

Minimum version of setuptools bumped to 64
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To comply with requirements of ``setuptools_scm``, the minimum version of ``setuptools``
has been increased from 42 to 64.
