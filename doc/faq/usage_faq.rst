.. _usage-faq:

***************
Usage
***************

.. contents::
   :backlinks: none


.. _general_concepts:

General Concepts
================

:mod:`matplotlib` has an extensive codebase that can be daunting to many
new users. However, most of matplotlib can be understood with a fairly
simple conceptual framework and knowledge of a few important points.

Plotting requires action on a range of levels, from the most general
(e.g., 'contour this 2-D array') to the most specific (e.g., 'color
this screen pixel red'). The purpose of a plotting package is to assist
you in visualizing your data as easily as possible, with all the necessary
control -- that is, by using relatively high-level commands most of
the time, and still have the ability to use the low-level commands when
needed.

Therefore, everything in matplotlib is organized in a hierarchy. At the top
of the hierarchy is the matplotlib "state-machine environment" which is
provided by the :mod:`matplotlib.pyplot` module. At this level, simple
functions are used to add plot elements (lines, images, text, etc.) to
the current axes in the current figure.

.. note::
   Pyplot's state-machine environment behaves similarly to MATLAB and
   should be most familiar to users with MATLAB experience.

The next level down in the hierarchy is the first level of the object-oriented
interface, in which pyplot is used only for a few functions such as figure
creation, and the user explicitly creates and keeps track of the figure
and axes objects. At this level, the user uses pyplot to create figures,
and through those figures, one or more axes objects can be created. These
axes objects are then used for most plotting actions.

For even more control -- which is essential for things like embedding
matplotlib plots in GUI applications -- the pyplot level may be dropped
completely, leaving a purely object-oriented approach.

.. _pylab:

Matplotlib, pylab, and pyplot: how are they related?
====================================================

Matplotlib is the whole package; :mod:`pylab` is a module in matplotlib
that gets installed alongside :mod:`matplotlib`; and :mod:`matplotlib.pyplot`
is a module in matplotlib.

Pyplot provides the state-machine interface to the underlying plotting
library in matplotlib. This means that figures and axes are implicitly
and automatically created to achieve the desired plot. For example,
calling ``plot`` from pyplot will automatically create the necessary
figure and axes to achieve the desired plot. Setting a title will
then automatically set that title to the current axes object::

    import matplotlib.pyplot as plt

    plt.plot(range(10), range(10))
    plt.title("Simple Plot")
    plt.show()

Pylab combines the pyplot functionality (for plotting) with the numpy
functionality (for mathematics and for working with arrays)
in a single namespace, making that namespace
(or environment) even more MATLAB-like.
For example, one can call the `sin` and `cos` functions just like
you could in MATLAB, as well as having all the features of pyplot.

The pyplot interface is generally preferred for non-interactive plotting
(i.e., scripting). The pylab interface is convenient for interactive
calculations and plotting, as it minimizes typing. Note that this is
what you get if you use the *ipython* shell with the *-pylab* option,
which imports everything from pylab and makes plotting fully interactive.

.. _coding_styles:

Coding Styles
==================

When viewing this documentation and examples, you will find different
coding styles and usage patterns. These styles are perfectly valid
and have their pros and cons. Just about all of the examples can be
converted into another style and achieve the same results.
The only caveat is to avoid mixing the coding styles for your own code.

.. note::
   Developers for matplotlib have to follow a specific style and guidelines.
   See :ref:`developers-guide-index`.

Of the different styles, there are two that are officially supported.
Therefore, these are the preferred ways to use matplotlib.

For the preferred pyplot style, the imports at the top of your
scripts will typically be::

    import matplotlib.pyplot as plt
    import numpy as np

Then one calls, for example, np.arange, np.zeros, np.pi, plt.figure,
plt.plot, plt.show, etc. So, a simple example in this style would be::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

Note that this example used pyplot's state-machine to
automatically and implicitly create a figure and an axes. For full
control of your plots and more advanced usage, use the pyplot interface
for creating figures, and then use the object methods for the rest::

    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 10, 0.2)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()

Next, the same example using a pure MATLAB-style::

    from pylab import *
    x = arange(0, 10, 0.2)
    y = sin(x)
    plot(x, y)
    show()


So, why all the extra typing as one moves away from the pure
MATLAB-style?  For very simple things like this example, the only
advantage is academic: the wordier styles are more explicit, more
clear as to where things come from and what is going on.  For more
complicated applications, this explicitness and clarity becomes
increasingly valuable, and the richer and more complete object-oriented
interface will likely make the program easier to write and maintain.

.. _what-is-a-backend:

What is a backend?
==================

A lot of documentation on the website and in the mailing lists refers
to the "backend" and many new users are confused by this term.
matplotlib targets many different use cases and output formats.  Some
people use matplotlib interactively from the python shell and have
plotting windows pop up when they type commands.  Some people embed
matplotlib into graphical user interfaces like wxpython or pygtk to
build rich applications.  Others use matplotlib in batch scripts to
generate postscript images from some numerical simulations, and still
others in web application servers to dynamically serve up graphs.

To support all of these use cases, matplotlib can target different
outputs, and each of these capabilities is called a backend; the
"frontend" is the user facing code, ie the plotting code, whereas the
"backend" does all the hard work behind-the-scenes to make the figure.
There are two types of backends: user interface backends (for use in
pygtk, wxpython, tkinter, qt4, or macosx; also referred to as
"interactive backends") and hardcopy backends to make image files
(PNG, SVG, PDF, PS; also referred to as "non-interactive backends").

There are a two primary ways to configure your backend.  One is to set
the ``backend`` parameter in your ``matplotlibrc`` file (see
:ref:`customizing-matplotlib`)::

    backend : WXAgg   # use wxpython with antigrain (agg) rendering

The other is to use the matplotlib :func:`~matplotlib.use` directive::

    import matplotlib
    matplotlib.use('PS')   # generate postscript output by default

If you use the ``use`` directive, this must be done before importing
:mod:`matplotlib.pyplot` or :mod:`matplotlib.pylab`.

.. note::
   Backend name specifications are not case-sensitive; e.g., 'GTKAgg'
   and 'gtkagg' are equivalent.

With a typical installation of matplotlib, such as from a
binary installer or a linux distribution package, a good default
backend will already be set, allowing both interactive work and
plotting from scripts, with output to the screen and/or to
a file, so at least initially you will not need to use either of the
two methods given above.

If, however, you want to write graphical user interfaces, or a web
application server (:ref:`howto-webapp`), or need a better
understanding of what is going on, read on. To make things a little
more customizable for graphical user interfaces, matplotlib separates
the concept of the renderer (the thing that actually does the drawing)
from the canvas (the place where the drawing goes).  The canonical
renderer for user interfaces is ``Agg`` which uses the `Anti-Grain
Geometry`_ C++ library to make a raster (pixel) image of the figure.
All of the user interfaces except ``macosx`` can be used with
agg rendering, eg
``WXAgg``, ``GTKAgg``, ``QT4Agg``, ``TkAgg``.  In
addition, some of the user interfaces support other rendering engines.
For example, with GTK, you can also select GDK rendering (backend
``GTK``) or Cairo rendering (backend ``GTKCairo``).

For the rendering engines, one can also distinguish between `vector
<http://en.wikipedia.org/wiki/Vector_graphics>`_ or `raster
<http://en.wikipedia.org/wiki/Raster_graphics>`_ renderers.  Vector
graphics languages issue drawing commands like "draw a line from this
point to this point" and hence are scale free, and raster backends
generate a pixel representation of the line whose accuracy depends on a
DPI setting.

Here is a summary of the matplotlib renderers (there is an eponymous
backed for each; these are *non-interactive backends*, capable of
writing to a file):

=============   ============   ================================================
Renderer        Filetypes      Description
=============   ============   ================================================
:term:`AGG`     :term:`png`    :term:`raster graphics` -- high quality images
                               using the `Anti-Grain Geometry`_ engine
PS              :term:`ps`     :term:`vector graphics` -- Postscript_ output
                :term:`eps`
PDF             :term:`pdf`    :term:`vector graphics` --
                               `Portable Document Format`_
SVG             :term:`svg`    :term:`vector graphics` --
                               `Scalable Vector Graphics`_
:term:`Cairo`   :term:`png`    :term:`vector graphics` --
                :term:`ps`     `Cairo graphics`_
                :term:`pdf`
                :term:`svg`
                ...
:term:`GDK`     :term:`png`    :term:`raster graphics` --
                :term:`jpg`    the `Gimp Drawing Kit`_
                :term:`tiff`
                ...
=============   ============   ================================================

And here are the user interfaces and renderer combinations supported;
these are *interactive backends*, capable of displaying to the screen
and of using appropriate renderers from the table above to write to
a file:

============   ================================================================
Backend        Description
============   ================================================================
GTKAgg         Agg rendering to a :term:`GTK` 2.x canvas (requires PyGTK_)
GTK3Agg        Agg rendering to a :term:`GTK` 3.x canvas (requires PyGObject_)
GTK            GDK rendering to a :term:`GTK` 2.x canvas (not recommended)
               (requires PyGTK_)
GTKCairo       Cairo rendering to a :term:`GTK` 2.x canvas (requires PyGTK_
               and pycairo_)
GTK3Cairo      Cairo rendering to a :term:`GTK` 3.x canvas (requires PyGObject_
               and pycairo_)
WXAgg          Agg rendering to to a :term:`wxWidgets` canvas
               (requires wxPython_)
WX             Native :term:`wxWidgets` drawing to a :term:`wxWidgets` Canvas
               (not recommended) (requires wxPython_)
TkAgg          Agg rendering to a :term:`Tk` canvas (requires TkInter_)
Qt4Agg         Agg rendering to a :term:`Qt4` canvas (requires PyQt4_)
macosx         Cocoa rendering in OSX windows
               (presently lacks blocking show() behavior when matplotlib
               is in non-interactive mode)
============   ================================================================

.. _`Anti-Grain Geometry`: http://www.antigrain.com/
.. _Postscript: http://en.wikipedia.org/wiki/PostScript
.. _`Portable Document Format`: http://en.wikipedia.org/wiki/Portable_Document_Format
.. _`Scalable Vector Graphics`: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _`Cairo graphics`: http://en.wikipedia.org/wiki/Cairo_(graphics)
.. _`Gimp Drawing Kit`: http://en.wikipedia.org/wiki/GDK
.. _PyGTK: http://www.pygtk.org
.. _PyGObject: https://live.gnome.org/PyGObject
.. _pycairo: http://www.cairographics.org/pycairo/
.. _wxPython: http://www.wxpython.org/
.. _TkInter: http://wiki.python.org/moin/TkInter
.. _PyQt4: http://www.riverbankcomputing.co.uk/software/pyqt/intro



.. _interactive-mode:

What is interactive mode?
===================================

Use of an interactive backend (see :ref:`what-is-a-backend`)
permits--but does not by itself require or ensure--plotting
to the screen.  Whether and when plotting to the screen occurs,
and whether a script or shell session continues after a plot
is drawn on the screen, depends on the functions and methods
that are called, and on a state variable that determines whether
matplotlib is in "interactive mode".  The default Boolean value is set
by the :file:`matplotlibrc` file, and may be customized like any other
configuration parameter (see :ref:`customizing-matplotlib`).  It
may also be set via :func:`matplotlib.interactive`, and its
value may be queried via :func:`matplotlib.is_interactive`.  Turning
interactive mode on and off in the middle of a stream of plotting
commands, whether in a script or in a shell, is rarely needed
and potentially confusing, so in the following we will assume all
plotting is done with interactive mode either on or off.

.. note::
   Major changes related to interactivity, and in particular the
   role and behavior of :func:`~matplotlib.pyplot.show`, were made in the
   transition to matplotlib version 1.0, and bugs were fixed in
   1.0.1.  Here we describe the version 1.0.1 behavior for the
   primary interactive backends, with the partial exception of
   *macosx*.

Interactive mode may also be turned on via :func:`matplotlib.pyplot.ion`,
and turned off via :func:`matplotlib.pyplot.ioff`.

.. note::
   Interactive mode works with suitable backends in ipython and in
   the ordinary python shell, but it does *not* work in the IDLE IDE.


Interactive example
--------------------

From an ordinary python prompt, or after invoking ipython with no options,
try this::

    import matplotlib.pyplot as plt
    plt.ion()
    plt.plot([1.6, 2.7])

Assuming you are running version 1.0.1 or higher, and you have
an interactive backend installed and selected by default, you should
see a plot, and your terminal prompt should also be active; you
can type additional commands such as::

    plt.title("interactive test")
    plt.xlabel("index")

and you will see the plot being updated after each line.  This is
because you are in interactive mode *and* you are using pyplot
functions.  Now try an alternative method of modifying the
plot.  Get a
reference to the :class:`~matplotlib.axes.Axes` instance, and
call a method of that instance::

    ax = plt.gca()
    ax.plot([3.1, 2.2])

Nothing changed, because the Axes methods do not include an
automatic call to :func:`~matplotlib.pyplot.draw_if_interactive`;
that call is added by the pyplot functions.  If you are using
methods, then when you want to update the plot on the screen,
you need to call :func:`~matplotlib.pyplot.draw`::

    plt.draw()

Now you should see the new line added to the plot.

Non-interactive example
-----------------------

Start a fresh session as in the previous example, but now
turn interactive mode off::

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.plot([1.6, 2.7])

Nothing happened--or at least nothing has shown up on the
screen (unless you are using *macosx* backend, which is
anomalous).  To make the plot appear, you need to do this::

    plt.show()

Now you see the plot, but your terminal command line is
unresponsive; the :func:`show()` command *blocks* the input
of additional commands until you manually kill the plot
window.

What good is this--being forced to use a blocking function?
Suppose you need a script that plots the contents of a file
to the screen.  You want to look at that plot, and then end
the script.  Without some blocking command such as show(), the
script would flash up the plot and then end immediately,
leaving nothing on the screen.

In addition, non-interactive mode delays all drawing until
show() is called; this is more efficient than redrawing
the plot each time a line in the script adds a new feature.

Prior to version 1.0, show() generally could not be called
more than once in a single script (although sometimes one
could get away with it); for version 1.0.1 and above, this
restriction is lifted, so one can write a script like this::

    import numpy as np
    import matplotlib.pyplot as plt
    plt.ioff()
    for i in range(3):
        plt.plot(np.random.rand(10))
        plt.show()

which makes three plots, one at a time.

Summary
-------

In interactive mode, pyplot functions automatically draw
to the screen.

When plotting interactively, if using
object method calls in addition to pyplot functions, then
call :func:`~matplotlib.pyplot.draw` whenever you want to
refresh the plot.

Use non-interactive mode in scripts in which you want to
generate one or more figures and display them before ending
or generating a new set of figures.  In that case, use
:func:`~matplotlib.pyplot.show` to display the figure(s) and
to block execution until you have manually destroyed them.
