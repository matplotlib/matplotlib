.. _colors:

*****************
Specifying Colors
*****************

In almost all places in matplotlib where a color can be specified by the user it can be provided as:

* ``(r, g, b)`` tuples
* ``(r, g, b, a)`` tuples
* hex string, ex ``#OFOFOF``
* float value between [0, 1] for gray level
* valid css4/X11 name
* valid name from the `XKCD color survey <http://blog.xkcd.com/2010/05/03/color-survey-results/>`__  This names maybe
  prefixed by ``'XKCD'`` as the css4 names have priority
* One of ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``

Internally, we are moving to storing all colors as RGBA quadruples.

Name clash between CSS4/X11 and XKCD
------------------------------------
