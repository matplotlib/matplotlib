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
  names are available both with and with out spaces.  In the case of name clashes
  the css/X11 names have priority.  To ensure colors
  from the XKCD mapping are used prefix the space-less name with
  ``'XKCD'``.

All string specifications of color are case-insensitive.

Internally, mpl is moving to storing all colors as RGBA float quadruples.

Name clash between CSS4/X11 and XKCD
------------------------------------

The color names in the XKCD survey include spaces (unlike css4/X11
names).  Matplotlib exposes all of the XKCD colors both with and
without spaces.

There are 95 (out of 148 colors in the css color list) conflicts
between the css4/X11 names and the XKCD names.  Given that these are
the standard color names of the web, matplotlib should follow these
conventions.  To accesses the XKCD colors which are shadowed by css4,
prefix the colorname with ``'XKCD'``, for example ``'blue'`` maps to
``'#0000FF'`` where as ``'XKCDblue'`` maps to ``'#0343DF'``.

.. plot::

   import matplotlib.pyplot as plt
   import matplotlib._color_data as mcd

   import matplotlib.patches as mpatch
   overlap = (set(mcd.CSS4_COLORS) & set(mcd.XKCD_COLORS))

   fig = plt.figure(figsize=[4.8, 16])
   ax = fig.add_axes([0, 0, 1, 1])

   j = 0

   for n in sorted(overlap, reverse=True):
       cn = mcd.CSS4_COLORS[n]
       xkcd = mcd.XKCD_COLORS[n].upper()
       if cn != xkcd:
           print (n, cn, xkcd)

       r1 = mpatch.Rectangle((0, j), 1, 1, color=cn)
       r2 = mpatch.Rectangle((1, j), 1, 1, color=xkcd)
       txt = ax.text(2, j+.5, '  ' + n, va='center', fontsize=10)
       ax.add_patch(r1)
       ax.add_patch(r2)
       ax.axhline(j, color='k')
       j += 1

   ax.text(.5, j+.1, 'X11', ha='center')
   ax.text(1.5, j+.1, 'XKCD', ha='center')
   ax.set_xlim(0, 3)
   ax.set_ylim(0, j + 1)
   ax.axis('off')
