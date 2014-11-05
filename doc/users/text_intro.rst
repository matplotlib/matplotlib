.. _text-intro:

Text introduction
=================

matplotlib has excellent text support, including mathematical
expressions, truetype support for raster and vector outputs, newline
separated text with arbitrary rotations, and unicode support.  Because
we embed the fonts directly in the output documents, e.g., for postscript
or PDF, what you see on the screen is what you get in the hardcopy.
`freetype2 <http://freetype.sourceforge.net/index2.html>`_ support
produces very nice, antialiased fonts, that look good even at small
raster sizes.  matplotlib includes its own
:mod:`matplotlib.font_manager`, thanks to Paul Barrett, which
implements a cross platform, W3C compliant font finding algorithm.

You have total control over every text property (font size, font
weight, text location and color, etc) with sensible defaults set in
the rc file.  And significantly for those interested in mathematical
or scientific figures, matplotlib implements a large number of TeX
math symbols and commands, to support :ref:`mathematical expressions
<mathtext-tutorial>` anywhere in your figure.


Basic text commands
===================

The following commands are used to create text in the pyplot
interface

* :func:`~matplotlib.pyplot.text` - add text at an arbitrary location to the ``Axes``;
  :meth:`matplotlib.axes.Axes.text` in the API.

* :func:`~matplotlib.pyplot.xlabel` - add an axis label to the x-axis;
  :meth:`matplotlib.axes.Axes.set_xlabel` in the API.

* :func:`~matplotlib.pyplot.ylabel` - add an axis label to the y-axis;
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

.. plot:: pyplots/text_commands.py
   :include-source:
