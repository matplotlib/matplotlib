.. _colors:

*****************
Specifying Colors
*****************

In almost all places in matplotlib where a color can be specified by the user
it can be provided as:

* ``(r, g, b)`` tuples
* ``(r, g, b, a)`` tuples
* hex string, ex ``#0F0F0F``, or ``#0F0F0F0F`` (with alpha channel)
* float value between [0, 1] for gray level
* One of ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``
* valid CSS4/X11 color names
* valid name from the `xkcd color survey
  <http://blog.xkcd.com/2010/05/03/color-survey-results/>`__ These
  names are prefixed with ``'xkcd:'`` (e.g., ``'xkcd:sky blue'``) to
  prevent name clashes with the CSS4/X11 names.

All string specifications of color are case-insensitive.

Internally, mpl is moving to storing all colors as RGBA float quadruples.

There are 95 (out of 148 colors in the css color list) conflicts between the
CSS4/X11 names and the xkcd names.  Given that the former are the standard
color names of the web, matplotlib should follow them.  Thus, xkcd color names
are prefixed with ``'xkcd:'``, for example ``'blue'`` maps to ``'#0000FF'``
where as ``'xkcd:blue'`` maps to ``'#0343DF'``.

.. plot::

   import matplotlib.pyplot as plt
   import matplotlib._color_data as mcd
   import matplotlib.patches as mpatch

   overlap = {name for name in mcd.CSS4_COLORS
              if "xkcd:" + name in mcd.XKCD_COLORS}

   fig = plt.figure(figsize=[4.8, 16])
   ax = fig.add_axes([0, 0, 1, 1])

   for j, n in enumerate(sorted(overlap, reverse=True)):
       cn = mcd.CSS4_COLORS[n]
       xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
       if cn != xkcd:
           print(n, cn, xkcd)

       r1 = mpatch.Rectangle((0, j), 1, 1, color=cn)
       r2 = mpatch.Rectangle((1, j), 1, 1, color=xkcd)
       txt = ax.text(2, j+.5, '  ' + n, va='center', fontsize=10)
       ax.add_patch(r1)
       ax.add_patch(r2)
       ax.axhline(j, color='k')

   ax.text(.5, j + .1, 'X11', ha='center')
   ax.text(1.5, j + .1, 'XKCD', ha='center')
   ax.set_xlim(0, 3)
   ax.set_ylim(0, j + 1)
   ax.axis('off')
