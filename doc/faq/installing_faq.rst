.. _installing-faq:

******************
 Installation FAQ
******************



How do I report a compilation problem?
======================================

See :ref:`reporting-problems`.

.. _clean-install:

How do I cleanly rebuild and reinstall everything?
==================================================

Unfortunately::

    python setup.py clean

does not properly clean the build directory, and does nothing to the
install directory.  To cleanly rebuild:

    * delete the ``build`` directory in the source tree
    * delete ``site-packages/matplotlib`` directory in the Python
      installation.  The location of ``site-packages`` is
      platform-specific.  You can find out where matplotlib is installed by doing::

          > python -c 'import matplotlib; print matplotlib.__file__'

      and then making sure you remove the matplotlib directory (and
      any matplotlib*.egg files) you find there.

    * you may also want to clear some of the cache data that
      matplotlib stores in your ``.matplotlib`` directory.  You can
      find the location of this directory by doing::

          import matplotlib
          print matplotlib.get_configdir()

      A typical location for the config directory is :file:`.matplotlib`, and the following
      caches may need to be cleared after a major update::

          rm -rf ~/.matplotlib/tex.cache
	  rm -rf ~/.matplotlib/fontManager.cache

.. _what-is-a-backend:

What is a backend?
==================

A lot of documentation on the website and in the mailing lists refers
to the "backend" and many new users are confused by this term.
matplotlib targets many different use cases and output formats.  Some
people use matplotlib interactively from the python shell and have
plotting windows pop up when they type commands.  Some people embed
matplotlib into graphical user interfaces like wxpython or pygtk to
build rich applications.  Others use matplotlib in batch scripts, to
generate postscript images from some numerical simulations, and still
others in web application servers to dynamically serve up graphs.

To support all of these use cases, matplotlib can target different
outputs, and each of these capabililities is called a backend (the
"frontend" is the user facing code, ie the plotting code, whereas the
"backend" does all the dirty work behind the scenes to make the
figure.  There are two types of backends: user interface backends (for
use in pygtk, wxpython, tkinter, qt or fltk) and hardcopy backends to
make image files (PNG, SVG, PDF, PS).

There are a two primary ways to configure your backend.  One is to set
the ``backend`` parameter in you ``matplotlibrc`` file (see
:ref:`customizing-matplotlib`)::

    backend : WXAgg   # use wxpython with antigrain (agg) rendering

The other is to use the matplotlib :func:`~matplotlib.use` directive::

    import matplotlib
    matplotlib.use('PS')   # generate postscript output by default

If you use the ``use`` directive, this must be done before importing
:mod:`matplotlib.pyplot` or :mod:`matplotlib.pylab`.

If you are unsure what to do, and just want to get cranking, just set
your backend to ``TkAgg``.  This will do the right thing for 95% of the
users.  It gives you the option of running your scripts in batch or
working interactively from the python shell, with the least amount of
hassles, and is smart enough to do the right thing when you ask for
postscript, or pdf, or other image formats.

If however, you want to write graphical user interfaces, or a web
application server (:ref:`howto-webapp`), or need a better
understanding of what is going on, read on. To make things a little
more customizable for graphical user interfaces, matplotlib separates
the concept of the renderer (the thing that actually does the drawing)
from the canvas (the place where the drawing goes).  The canonical
renderer for user interfaces is ``Agg`` which uses the `antigrain
<http://antigrain.html>`_ C++ library to make a raster (pixel) image
of the figure.  All of the user interfaces can be used with agg
rendering, eg ``WXAgg``, ``GTKAgg``, ``QTAgg``, ``TkAgg``.  In
addition, some of the user interfaces support other rendering engines.
For example, with GTK, you can also select GDK rendering (backend
``GTK``) or Cairo rendering (backend ``GTKCairo``).

For the rendering engines, one can also distinguish between vector or
raster renderers.  Vector issue drawing commands like "draw a line
from this point to this point" and hence are scale free, and raster
backends generate a pixel represenation of the line whose accuracy
depends on a DPI setting.

Here is a summary of the matplotlib renderers (there is an eponymous
backed for each):

===============================   =====================================================================================
Renderer (Filetypes)              Description
===============================   =====================================================================================
Agg (png)                         raster - high quality images using the `antigrain <http://antigrain.html>`_  engine
PS  (ps, eps)                     vector - postscript output
PDF (pdf)                         vector - portable document format
SVG (svg)                         vector - scalar vector graphics
Cairo (png, ps, pdf, svn, ...)    vector - `cairo graphics <http://cairographics.org>`_
GDK (png, jpg, tiff..)            raster - the GDK drawing API for GTK
===============================   =====================================================================================

And here are the user interfaces and renderer combinations supported:

============   ===================================================================================================
Backend        Description
============   ===================================================================================================
GTKAgg         Agg rendering to a GTK canvas (`pygtk <http://www.pygtk.org>`_)
GTK            GDK rendering to a GTK canvas (not recommended) (`pygtk <http://www.pygtk.org>`_)
GTKCairo       Cairo rendering to a GTK Canvas (`pygtk <http://www.pygtk.org>`_)
WXAgg          Agg rendering to to a WX canvas (`wxpython <http://www.wxpython.org>`_)
WX             Native WX drawing to a WX Canvas (not recommended) (`wxpython <http://www.wxpython.org>`_)
TkAgg          Agg rendering to a Tkinter canvas (`tkinter <http://wiki.python.org/moin/TkInter>`_)
QtAgg          Agg rendering to a Qt canvas (`pyqt <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_)
Qt4Agg         Agg rendering to a Qt4 canvas (`pyqt <http://www.riverbankcomputing.co.uk/software/pyqt/intro>`_)
FLTKAgg        Agg rendering to a FLTK canvas (`pyfltk <http://pyfltk.sourceforge.net>`_)
============   ===================================================================================================


OS-X questions
==============

.. _easy-install-osx-egg:

How can I easy_install my egg?
------------------------------

I downloaded the egg for 0.98 from the matplotlib webpages,
and I am trying to ``easy_install`` it, but I am getting an error::

    > easy_install ./matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg
    Processing matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg
    removing '/Library/Python/2.5/site-packages/matplotlib-0.98.0-py2.5-
    ...snip...
    Reading http://matplotlib.sourceforge.net
    Reading http://cheeseshop.python.org/pypi/matplotlib/0.91.3
    No local packages or download links found for matplotlib==0.98.0
    error: Could not find suitable distribution for
    Requirement.parse('matplotlib==0.98.0')

If you rename ``matplotlib-0.98.0-py2.5-macosx-10.3-fat.egg`` to
``matplotlib-0.98.0-py2.5.egg``, ``easy_install`` will install it from
the disk.  Many Mac OS X eggs with cruft at the end of the filename,
which prevents their installation through easy_install.  Renaming is
all it takes to install them; still, it's annoying.

Windows questions
=================

.. _windows-installers:

Where can I get binary installers for windows?
----------------------------------------------

If you have already installed python, you can use one of the
matplotlib binary installers for windows -- you can get these from the
`sourceforge download
<http://sourceforge.net/project/platformdownload.php?group_id=80706>`_
site.  Choose the files that match your version of python (eg
``py2.5`` if you installed Python 2.5) which have the ``exe``
extension.  If you haven't already installed python, you can get the
official version from the `python web site
<http://python.org/download/>`_.  There are also two packaged
distributions of python that come preloaded with matplotlib and many
other tools like ipython, numpy, scipy, vtk and user interface
toolkits.  These packages are quite large because they come with so
much, but you get everything with a single click installer.

* the enthought python distribution `EPD
  <http://www.enthought.com/products/epd.php>`_

* `python (x, y) <http://www.pythonxy.com/foreword.php>`_
