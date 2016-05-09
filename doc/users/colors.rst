.. _colors:

*****************
Specifying Colors
*****************

In almost all places in matplotlib where a color can be specified by the user it can be provided as:

* ``(r, g, b)`` tuples
* ``(r, g, b, a)`` tuples
* hex string, ex ``#OFOFOF``
* float value between [0, 1] for gray level
* One of ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``
* valid css4/X11 color names
* valid name from the `XKCD color survey
  <http://blog.xkcd.com/2010/05/03/color-survey-results/>`__ These
  names are prefixed with ``'xkcd:'`` (e.g., ``'xkcd:sky blue'``) to
  prevent name clashes with the CSS4/X11 names.

All string specifications of color are case-insensitive.

Internally, mpl is moving to storing all colors as RGBA float quadruples.
