***************
pyplot tutorial
***************

:mod:`matplotlib.pyplot` is a collection of functions that make
matplotlib work like matlab.  Each ``pyplot`` function makes some
change to a figure: eg, create a figure, create a plotting area in a
figure, plot some lines in a plotting area, decorate the plot with
labels, etc....  :mod:`matplotlib.pyplot` is stateful, in that it
keeps track of the current figure and plotting area, and the plotting
functions are directed to the current axes

.. literalinclude:: figures/pyplot_simple.py

.. image:: figures/pyplot_simple.png
   :scale: 50


You may be wondering why the x-axis ranges from 0-3 and the y-axis
from 1-4.  If you provide a single list or array to the
:func:`~matplotlib.pyplot.plot` command, matplotlib assumes it a
sequence of y values, and automatically generates the x values for
you.  Since python ranges start with 0, the default x vector has the
same length as y but starts with 0.  Hence the x data are
``[0,1,2,3]``.

:func:`~matplotlib.pyplot.plot` is a versatile command, and will take
an arbitrary number of arguments.  For example, to plot x versus y,
you can issue the command::
    
    plt.plot([1,2,3,4], [1,4,9,16])

For every x, y pair of arguments, there is a optional third argument
which is the format string that indicates the color and line type of
the plot.  The letters and symbols of the format string are from
matlab, and you concatenate a color string with a line style string.
The default format string is 'b-', which is a solid blue line.  For
example, to plot the above with red circles, you would issue

.. literalinclude:: figures/pyplot_formatstr.py

.. image:: figures/pyplot_formatstr.png
   :scale: 50

See the :func:`~matplotlib.pyplot.plot` documentation for a complete
list of line styles and format strings.  The
:func:`~matplotlib.pyplot.axis` command in the example above takes a
list of ``[xmin, xmax, ymin, ymax]`` and specifies the viewport of the
axes.

If matplotlib were limited to working with lists, it would be fairly
useless for numeric processing.  Generally, you will use `numpy
<http://numpy.scipy.org>`_ arrays.  In fact, all sequences are
converted to numpy arrays internally.  The example below illustrates a
plotting several lines with different format styles in one command
using arrays.

.. literalinclude:: figures/pyplot_three.py

.. image:: figures/pyplot_three.png
   :scale: 50



Controlling line properties
===========================

Lines have many attributes that you can set: linewidth, dash style,
antialiased, etc; see :class:`matplotlib.lines.Line2D`.  There are
several ways to set line properties

* Use keyword args::
 
      plt.plot(x, y, linewidth=2.0)


* Use the setter methods of the ``Line2D`` instance.  ``plot`` returns a list
  of lines; eg ``line1, line2 = plot(x1,y1,x2,x2)``.  Below I have only
  one line so it is a list of length 1.  I use tuple unpacking in the
  ``line, = plot(x, y, 'o')`` to get the first element of the list::

      line, = plt.plot(x, y, 'o')
      line.set_antialiased(False) # turn off antialising

* Use the :func:`~matplotlib.pyplot.setp` command.  The example below
  uses matlab handle graphics style command to set multiple properties
  on a list of lines.  ``setp`` works transparently with a list of objects
  or a single object.  You can either use python keyword arguments or
  matlab-style string/value pairs::

      lines = plt.plot(x1, y1, x2, y2)
      # use keyword args
      plt.setp(lines, color='r', linewidth=2.0)
      # or matlab style string value pairs
      plt.setp(lines, 'color', 'r', 'linewidth', 2.0)


Here are the available :class:`~matplotlib.lines.Line2D` properties.

======================  ==================================================
Property                Value Type
======================  ==================================================
alpha			float
animated		[True | False]
antialiased or aa	[True | False]
clip_box		a matplotlib.transform.Bbox instance
clip_on			[True | False]
clip_path		a Path instance and a Transform instance, a Patch
color or c		any matplotlib color
contains		the hit testing function
dash_capstyle		['butt' | 'round' | 'projecting']
dash_joinstyle		['miter' | 'round' | 'bevel']
dashes			sequence of on/off ink in points
data			(np.array xdata, np.array ydata)
figure			a matplotlib.figure.Figure instance
label			any string
linestyle or ls		[ '-' | '--' | '-.' | ':' | 'steps' | ...]
linewidth or lw		float value in points
lod			[True | False]
marker			[ '+' | ',' | '.' | '1' | '2' | '3' | '4'
markeredgecolor or mec	any matplotlib color
markeredgewidth or mew	float value in points
markerfacecolor or mfc	any matplotlib color
markersize or ms	float
picker			used in interactive line selection
pickradius		the line pick selection radius
solid_capstyle		['butt' | 'round' |  'projecting']
solid_joinstyle		['miter' | 'round' | 'bevel']
transform		a matplotlib.transforms.Transform instance
visible			[True | False]
xdata			np.array
ydata			np.array
zorder			any number
======================  ==================================================

To get a list of settable line properties, call the
:func:`~matplotlib.pyplot.setp` function with a line or lines
as argument

.. sourcecode:: ipython

    In [69]: lines = plot([1,2,3])

    In [70]: setp(lines)
      alpha: float
      animated: [True | False]
      antialiased or aa: [True | False]
      ...snip

Working with multiple figure and axes
=====================================


Matlab, and :mod:`~matplotlib.pyplot`, have the concept of the current
figure and the current axes.  All plotting commands apply to the
current axes.  The function :func:`~matplotlib.pyplot.gca` returns the
current axes (a :class:`matplotlib.axes.Axes` instance), and
:func:`~matplotlib.pyplot.gcf` returns the current figure
(:class:`matplotlib.figure.Figure` instance). Normally, you don't have
to worry about this, because it is all taken care of behind the
scenes.  Below is an script to create two subplots.

.. literalinclude:: figures/pyplot_two_subplots.py

.. image:: figures/pyplot_two_subplots.png
   :scale: 50

The :func:`~matplotlib.pyplot.figure` command here is optional because
``figure(1)`` will be created by default, just as a ``subplot(111)``
will be created by default if you don't manually specify an axes.  The
:func:`~matplotlib.pyplot.subplot` command specifies ``numrows,
numcols, fignum`` where ``fignum`` ranges from 1 to
``numrows*numcols``.  The commas in the ``subplot command are optional
if ``numrows*numcols<10``.  So ``subplot(211)`` is identical to
``subplot(2,1,1)``.  You can create an arbitrary number of subplots
and axes.  If you want to place an axes manually, ie, not on a
rectangular grid, use the :func:`~matplotlib.pyplot.axes` command,
which allows you to specify the location as ``axes([left, bottom,
width, height])`` where all values are in fractional (0 to 1)
coordinates.  See `axes_demo.py
<http://matplotlib.sf.net/examples/axes_demo.py>`_ for an example of
placing axes manually and `line_styles.py
<http://matplotlib.sf.net/examples/line_styles.py>`_ for an example
with lots-o-subplots.


You can create multiple figures by using multiple
:func:`~matplotlib.pyplot.figure` calls with an increasing figure
number.  Of course, each figure can contain as many axes and subplots
as your heart desires::

    import matplotlib.pyplot as plt
    plt.figure(1)                # the first figure
    plt.plot([1,2,3])

    plt.figure(2)                # a second figure
    plt.plot([4,5,6])

    plt.figure(1)                # figure 1 current
    plt.title('Easy as 1,2,3')   # figure 1 title

You can clear the current figure with :func:`~matplotlib.pyplot.clf`
and the current axes with :func:`~matplotlib.pyplot.cla`.

Working with text
=================

The :func:`~matplotlib.pyplot.text` command can be used to add text in
an arbitrary location, and the :func:`~matplotlib.pyplot.xlabel`,
:func:`~matplotlib.pyplot.ylabel` and :func:`~matplotlib.pyplot.title`
are used to add text in the indicated locations. 

.. literalinclude:: figures/pyplot_text.py

.. image:: figures/pyplot_text.png
   :scale: 50


All of the text commands The :func:`~matplotlib.pyplot.text` commands
return an :class:`matplotlib.text.Text` instance.  Just as with with
lines above, you can customize the properties by passing keyword
arguments into the text functions or using
:func:`~matplotlib.pyplot.setp`::

  t = plt.xlabel('my data', fontsize=14, color='red')

The following font properties can be set

==========================  ==============================================================================
Property                    Value Type
==========================  ==============================================================================
alpha			    float
backgroundcolor		    any matplotlib color
bbox			    rectangle prop dict plus key 'pad' which is a pad in points
clip_box		    a matplotlib.transform.Bbox instance
clip_on			    [True | False]
clip_path		    a Path instance and a Transform instance, a Patch
color			    any matplotlib color
family			    [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
fontproperties		    a matplotlib.font_manager.FontProperties instance
horizontalalignment or ha   [ 'center' | 'right' | 'left' ]
label			    any string
linespacing		    float
multialignment		    ['left' | 'right' | 'center' ]
name or fontname	    string eg, ['Sans' | 'Courier' | 'Helvetica' ...]
picker			    [None|float|boolean|callable]
position		    (x,y)
rotation		    [ angle in degrees 'vertical' | 'horizontal'
size or fontsize	    [ size in points | relative size eg 'smaller', 'x-large' ]
style or fontstyle	    [ 'normal' | 'italic' | 'oblique']
text			    string or anything printable with '%s' conversion
transform		    a matplotlib.transform transformation instance
variant			    [ 'normal' | 'small-caps' ]
verticalalignment or va	    [ 'center' | 'top' | 'bottom' | 'baseline' ]
visible			    [True | False]
weight or fontweight	    [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
x			    float
y			    float
zorder			    any number
==========================  ==============================================================================


See `align_text <http://matplotlib.sf.net/screenshots.html#align_text>`_ for
examples of how to control the alignment and orientation of text.




Writing mathematical expressions
================================

You may have noticed in the histogram example above that we slipped a
little TeX markup into the expression ``r'$\mu=100,\ \sigma=15$')``
You can use TeX markup in any matplotlib text string; see the
:mod:`matplotlib.mathtext` module documentation for details.  Note
that you do not need to have TeX installed, since matplotlib ships
its own TeX expression parser, layout engine and fonts.  Michael
Droettboom has implemented the Knuth layout algorithms in python, so
the quality is quite good (matplotlib also provides a ``usetex`` option
for those who do want to call out to TeX to generate their text).

Any text element can use math text.  You need to use raw strings
(preceed the quotes with an ``'r'``), and surround the string text
with dollar signs, as in TeX.  Regular text and mathtext can be
interleaved within the same string.  Mathtext can use the Bakoma
Computer Modern fonts, STIX fonts or a Unicode font that you provide.
The mathtext font can be selected with the customization variable
``mathtext.fontset``::

    # plain text
    plt.title('alpha > beta')

    # math text
    plt.title(r'$\alpha > \beta$')


To make subscripts and superscripts use the '_' and '^' symbols::

    plt.title(r'$\alpha_i > \beta_i$')

You can also use a large number of the TeX symbols, as in ``\infty,
\leftarrow, \sum, \int``; see :class:`matplotlib.mathtext` for a
complete list.  The over/under subscript/superscript style is also
supported.  To write the sum of x_i from 0 to infinity, you could do::

    plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$')

The default font is *italics* for mathematical symbols.  To change
fonts, eg, to write "sin" in a Roman font, enclose the text in a font
command::

    plt.text(1,2, r's(t) = $\mathcal{A}\mathrm{sin}(2 \omega t)$')


Even better, many commonly used function names that are typeset in a
Roman font have shortcuts.  So the expression above could be written
as follows::

    plt.text(1,2, r's(t) = $\mathcal{A}\sin(2 \omega t)$')


Here "s" and "t" are variable in italics font (default), "sin" is in
Roman font, and the amplitude "A" is in caligraphy font. The font
choices are Roman ``\mathrm``, italics ``\mathit``, caligraphy
``\mathcal``, and typewriter ``\mathtt``.  If using the STIX fonts,
you also have the choice of blackboard (double-struck) ``\mathbb``,
circled ``\mathcircled``, Fraktur ``\mathfrak``, script (cursive)
``\mathscr`` and sans-serif ``\mathsf``.

The following accents are provided: ``\hat``, ``\breve``, ``\grave``,
``\bar``, ``\acute``, ``\tilde``, ``\vec``, ``\dot``, ``\ddot``.  All
of them have the same syntax, eg to make an overbar you do ``\bar{o}``
or to make an o umlaut you do ``\ddot{o}``.

.. literalinclude:: figures/pyplot_mathtext.py

.. image:: figures/pyplot_mathtext.png
   :scale: 50



    