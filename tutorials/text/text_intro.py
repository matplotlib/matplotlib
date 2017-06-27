"""
Text introduction
=================

Introduction to plotting and working with text in Matplotlib.

Matplotlib has extensive text support, including support for
mathematical expressions, truetype support for raster and
vector outputs, newline separated text with arbitrary
rotations, and unicode support.

Because it embeds fonts directly in output documents, e.g., for postscript
or PDF, what you see on the screen is what you get in the hardcopy.
`FreeType <https://www.freetype.org/>`_ support
produces very nice, antialiased fonts, that look good even at small
raster sizes.  matplotlib includes its own
:mod:`matplotlib.font_manager` (thanks to Paul Barrett), which
implements a cross platform, `W3C <http://www.w3.org/>`
compliant font finding algorithm.

The user has a great deal of control over text properties (font size, font
weight, text location and color, etc.) with sensible defaults set in
the `rc file <http://matplotlib.org/users/customizing.html>`.
And significantly, for those interested in mathematical
or scientific figures, matplotlib implements a large number of TeX
math symbols and commands, supporting :ref:`mathematical expressions
<sphx_glr_tutorials_text_mathtext.py>` anywhere in your figure.


Basic text commands
===================

The following commands are used to create text in the pyplot
interface

* :func:`~matplotlib.pyplot.text` - add text at an arbitrary location to the ``Axes``;
  :meth:`matplotlib.axes.Axes.text` in the API.

* :func:`~matplotlib.pyplot.xlabel` - add a label to the x-axis;
  :meth:`matplotlib.axes.Axes.set_xlabel` in the API.

* :func:`~matplotlib.pyplot.ylabel` - add a label to the y-axis;
  :meth:`matplotlib.axes.Axes.set_ylabel` in the API.

* :func:`~matplotlib.pyplot.title` - add a title to the ``Axes``;
  :meth:`matplotlib.axes.Axes.set_title` in the API.

* :func:`~matplotlib.pyplot.figtext` - add text at an arbitrary location to the ``Figure``;
  :meth:`matplotlib.figure.Figure.text` in the API.

* :func:`~matplotlib.pyplot.suptitle` - add a title to the ``Figure``;
  :meth:`matplotlib.figure.Figure.suptitle` in the API.

* :func:`~matplotlib.pyplot.annotate` - add an annotation, with
   optional arrow, to the ``Axes`` ; :meth:`matplotlib.axes.Axes.annotate`
   in the API.

All of these functions create and return a
:func:`matplotlib.text.Text` instance, which can be configured with a
variety of font and other properties.  The example below shows all of
these commands in action.
"""

import matplotlib.pyplot as plt

fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

plt.show()
