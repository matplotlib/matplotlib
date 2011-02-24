.. _installing-faq:

*****************
 Installation FAQ
*****************


.. contents::
   :backlinks: none


Report a compilation problem
======================================

See :ref:`reporting-problems`.

matplotlib compiled fine, but nothing shows up with plot
==========================================================

The first thing to try is a :ref:`clean install <clean-install>` and see if
that helps.  If not, the best way to test your install is by running a script,
rather than working interactively from a python shell or an integrated
development environment such as :program:`IDLE` which add additional
complexities. Open up a UNIX shell or a DOS command prompt and cd into a
directory containing a minimal example in a file. Something like
:file:`simple_plot.py`, or for example::

  from pylab import *
  plot([1,2,3])
  show()

and run it with::

  python simple_plot.py --verbose-helpful

This will give you additional information about which backends matplotlib is
loading, version information, and more. At this point you might want to make
sure you understand matplotlib's :ref:`configuration <customizing-matplotlib>`
process, governed by the :file:`matplotlibrc` configuration file which contains
instructions within and the concept of the matplotlib backend.

If you are still having trouble, see :ref:`reporting-problems`.

.. _clean-install:

Cleanly rebuild and reinstall everything
==================================================

The steps depend on your platform and installation method.

Easy Install
------------

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Run::

     easy_install -m PackageName


3. Delete any .egg files or directories from your :ref:`installation
   directory <locating-matplotlib-install>`.



Windows installer
-----------------

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Use :menuselection:`Start --> Control Panel` to start the :program:`Add and
   Remove Software` utility.

Source install
--------------

Unfortunately::

    python setup.py clean

does not properly clean the build directory, and does nothing to the
install directory.  To cleanly rebuild:

1. Delete the caches from your :ref:`.matplotlib configuration directory
   <locating-matplotlib-config-dir>`.

2. Delete the ``build`` directory in the source tree

3. Delete any matplotlib directories or eggs from your `installation directory
   <locating-matplotlib-install>`


.. _install-from-git:

Install from git
================

Clone the main source using one of::

   git clone git@github.com:matplotlib/matplotlib.git

or::

   git clone git://github.com/matplotlib/matplotlib.git

and build and install as usual with::

  > cd matplotlib
  > python setup.py install

If you want to be able to follow the development branch as it changes just replace
the last step with (Make sure you have **setuptools** installed)::

  > python setupegg.py develop

This creates links in the right places and installs the command line script to the appropriate places.
Then, if you want to update your **matplotlib** at any time, just do::

  > git pull

When you run `git pull`, if the output shows that only Python files have been
updated, you are all set. If C files have changed, you need to run the `python
setupegg.py develop` command again to compile them.

There is more information on :ref:`using git <using-git>` in
the developer docs.

Install from git
================

See :ref:`using-git`.

Backends
========

.. _what-is-a-backend:

What is a backend?
------------------

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
outputs, and each of these capabililities is called a backend; the
"frontend" is the user facing code, ie the plotting code, whereas the
"backend" does all the dirty work behind the scenes to make the
figure.  There are two types of backends: user interface backends (for
use in pygtk, wxpython, tkinter, qt, macosx, or fltk) and hardcopy backends to
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
rendering, eg ``WXAgg``, ``GTKAgg``, ``QTAgg``, ``TkAgg``,
``CocoaAgg``.  In addition, some of the user interfaces support other
rendering engines.  For example, with GTK, you can also select GDK
rendering (backend ``GTK``) or Cairo rendering (backend ``GTKCairo``).

For the rendering engines, one can also distinguish between `vector
<http://en.wikipedia.org/wiki/Vector_graphics>`_ or `raster
<http://en.wikipedia.org/wiki/Raster_graphics>`_ renderers.  Vector
graphics languages issue drawing commands like "draw a line from this
point to this point" and hence are scale free, and raster backends
generate a pixel represenation of the line whose accuracy depends on a
DPI setting.

Here is a summary of the matplotlib renderers (there is an eponymous
backed for each):

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

And here are the user interfaces and renderer combinations supported:

============   ================================================================
Backend        Description
============   ================================================================
GTKAgg         Agg rendering to a :term:`GTK` canvas (requires PyGTK_)
GTK            GDK rendering to a :term:`GTK` canvas (not recommended)
               (requires PyGTK_)
GTKCairo       Cairo rendering to a :term:`GTK` Canvas (requires PyGTK_)
WXAgg          Agg rendering to to a :term:`wxWidgets` canvas
               (requires wxPython_)
WX             Native :term:`wxWidgets` drawing to a :term:`wxWidgets` Canvas
               (not recommended) (requires wxPython_)
TkAgg          Agg rendering to a :term:`Tk` canvas (requires TkInter_)
QtAgg          Agg rendering to a :term:`Qt` canvas (requires PyQt_)
Qt4Agg         Agg rendering to a :term:`Qt4` canvas (requires PyQt4_)
FLTKAgg        Agg rendering to a :term:`FLTK` canvas (requires pyFLTK_)
macosx         Cocoa rendering in OSX windows
============   ================================================================

.. _`Anti-Grain Geometry`: http://www.antigrain.com/
.. _Postscript: http://en.wikipedia.org/wiki/PostScript
.. _`Portable Document Format`: http://en.wikipedia.org/wiki/Portable_Document_Format
.. _`Scalable Vector Graphics`: http://en.wikipedia.org/wiki/Scalable_Vector_Graphics
.. _`Cairo graphics`: http://en.wikipedia.org/wiki/Cairo_(graphics)
.. _`Gimp Drawing Kit`: http://en.wikipedia.org/wiki/GDK
.. _PyGTK: http://www.pygtk.org
.. _wxPython: http://www.wxpython.org/
.. _TkInter: http://wiki.python.org/moin/TkInter
.. _PyQt: http://www.riverbankcomputing.co.uk/software/pyqt/intro
.. _PyQt4: http://www.riverbankcomputing.co.uk/software/pyqt/intro
.. _pyFLTK: http://pyfltk.sourceforge.net


.. _pygtk-2.4:

Compile matplotlib with PyGTK-2.4
-------------------------------------------

There is a `bug in PyGTK-2.4`_. You need to edit
:file:`pygobject.h` to add the :cmacro:`G_BEGIN_DECLS` and :cmacro:`G_END_DECLS`
macros, and rename :cdata:`typename` parameter to :cdata:`typename_`::

  -                       const char *typename,
  +                       const char *typename_,

.. _`bug in PyGTK-2.4`: http://bugzilla.gnome.org/show_bug.cgi?id=155304


OS-X questions
==============

.. _which-python-for-osx:

Which python for OS X?
----------------------

Apple ships with its own python, many users have had trouble
with it so there are alternatives.  If it is feasible for you, we
recommend the enthought python distribution `EPD
<http://www.enthought.com/products/epd.php>`_ for OS X (which comes
with matplotlib and much more) or the
`MacPython <http://wiki.python.org/moin/MacPython/Leopard>`_ or the
official OS X version from `python.org
<http://www.python.org/download/>`_.


.. _install_osx_binaries:

Installing OSX binaries
-----------------------

If you want to install matplotlib from one of the binary installers we
build, you have two choices: a mpkg installer, which is a typical
Installer.app, or an binary OSX egg, which you can install via
setuptools easy_install.

The mkpg installer will have a "zip" extension, and will have a name
like file:`matplotlib-0.99.0.rc1-py2.5-macosx10.5_mpkg.zip` depending on
the python, matplotlib, and OSX versions.  You need to unzip this file
using either the "unzip" command on OSX, or simply double clicking on
it to run StuffIt Expander.  When you double click on the resultant
mpkd directory, which will have a name like
file:`matplotlib-0.99.0.rc1-py2.5-macosx10.5.mpkg`, it will run the
Installer.app, prompt you for a password if you need system wide
installation privileges, and install to a directory like
file:`/Library/Python/2.5/site-packages/`, again depedending on your
python version.  This directory may not be in your python path, so you
can test your installation with::

  > python -c 'import matplotlib; print matplotlib.__version__, matplotlib.__file__'

If you get an error like::

    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: No module named matplotlib

then you will need to set your PYTHONPATH, eg::

    export PYTHONPATH=/Library/Python/2.5/site-packages:$PYTHONPATH

See also ref:`environment-variables`.

.. _easy-install-osx-egg:

easy_install from egg
------------------------------

You can also us the eggs we build for OSX (see the `installation
instructions
<http://pypi.python.org/pypi/setuptools#cygwin-mac-os-x-linux-other>`_
for easy_install if you do not have it on your system already).  You
can try::

    > easy_install matplotlib

which should grab the latest egg from the sourceforge site, but the
naming conventions for OSX eggs appear to be broken (see below) so
there is no guarantee the right egg will be found.  We recommend you
download the latest egg from our `download site
<http://sourceforge.net/projects/matplotlib/files/>`_ directly to your
harddrive, and manually install it with

    > easy_install --install-dir=~/dev/lib/python2.5/site-packages/  matplotlib-0.99.0.rc1-py2.5-macosx-10.5-i386.egg


Some users have reported problems with the egg for 0.98 from the
matplotlib download site, with ``easy_install``, getting an error::

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


.. _install_from_source_on_osx_epd:

Building and installing from source on OSX with EPD
---------------------------------------------------

If you have the EPD installed (:ref:`which-python-for-osx`), it might turn out
to be rather tricky to install a new version of matplotlib from source on the
Mac OS 10.5 . Here's a procedure that seems to work, at least sometimes:

0. Remove the ~/.matplotlib folder ("rm -rf ~/.matplotlib").

1. Edit the file (make a backup before you start, just in case):
``/Library/Frameworks/Python.framework/Versions/Current/lib/python2.5/config/Makefile``,
removing all occurrences of the string ``-arch ppc``, changing the line
``MACOSX_DEPLOYMENT_TARGET=10.3`` to ``MACOSX_DEPLOYMENT_TARGET=10.5`` and
changing the occurrences of ``MacOSX10.4u.sdk`` into ``MacOSX10.5.sdk``

2.  In
``/Library/Frameworks/Python.framework/Versions/Current/lib/pythonX.Y/site-packages/easy-install.pth``,
(where X.Y is the version of Python you are building against)
Comment out the line containing the name of the directory in which the
previous version of MPL was installed (Looks something like
``./matplotlib-0.98.5.2n2-py2.5-macosx-10.3-fat.egg``).

3. Save the following as a shell script , for example
``./install-matplotlib-epd-osx.sh``::

   NAME=matplotlib
   VERSION=v1.0.x
   PREFIX=$HOME
   #branch="release"
   branch="trunk"
   git clone git://github.com/matplotlib/matplotlib.git
   cd matplotlib
   if [ $branch = "release" ]
       then
       echo getting the maintenance branch
       git checkout -b $VERSION origin/$VERSION
   fi
   export CFLAGS="-Os -arch i386"
   export LDFLAGS="-Os -arch i386"
   export PKG_CONFIG_PATH="/usr/x11/lib/pkgconfig"
   export ARCHFLAGS="-arch i386"
   python setup.py build
   # use --prefix if you don't want it installed in the default location:
   python setup.py install #--prefix=$PREFIX
   cd ..

Run this script (for example ``sh ./install-matplotlib-epd-osx.sh``) in the
directory in which you want the source code to be placed, or simply type the
commands in the terminal command line. This script sets some local variable
(CFLAGS, LDFLAGS, PKG_CONFIG_PATH, ARCHFLAGS), removes previous installations,
checks out the source from github, builds and installs it. The backend seems
to be set to MacOSX.


Windows questions
=================

.. _windows-installers:

Binary installers for windows
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
