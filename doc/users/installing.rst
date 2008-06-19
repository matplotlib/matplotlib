.. _installing:

**********
Installing
**********

Dependencies
============

**Requirements**

These are external packages which you will need to install before
installing matplotlib. Windows users only need the first two (python
and numpy) since the others are built into the matplotlib windows
installers available for download at the sourceforge site.

python 2.4 (or later but not python3)
    matplotlib requires python 2.4 or later

numpy 1.1 (or later)
    array support for python

libpng 1.1 (or later)
    library for loading and saving PNG files. libpng requires zlib. If
    you are a windows user, you can ignore this since we build support
    into the matplotlib single click installer.

freetype 1.4 (or later)
    library for reading true type font files. If you are a windows
    user, you can ignore this since we build support into the
    matplotlib single click installer.

**Optional**

These are optional packages which you may want to install to use
matplotlib with a user interface toolkit. See
:ref:`what-is-a-backend` for more details on the optional matplotlib
backends and the capabilities they provide

tk 8.3 or later
    The TCL/Tk widgets library used by the TkAgg backend

pyqt 3.1 or later
    The Qt3 widgets library python wrappers for the QtAgg backend

pyqt 4.0 or later
    The Qt4 widgets library python wrappersfor the Qt4Agg backend

pygtk 2.2 or later
    The python wrappers for the GTK widgets library for use with the GTK or GTKAgg backend

wxpython 2.6 or later
    The python wrappers for the wx widgets library for use with the WXAgg backend

wxpython 2.8 or later
    The python wrappers for the wx widgets library for use with the WX backend

pyfltk 1.0 or later
    The python wrappers of the FLTK widgets library for use with FLTKAgg

**Required libraries that ship with matplotlib**

If you are downloading matplotlib or installing from source or
subversion, you can ignore this section. This is useful for matplotlib
developers and packagers who may want to disable the matplotlib
version and ship a packaged version.

agg2.4
    The antigrain C++ rendering engine

pytz 2007g or later
    timezone handling for python datetime objects

dateutil 1.1 or later
    extensions to python datetime handling

**Optional libraries that ship with matplotlib**

As above, if you are downloading matplotlib or installing from source
or subversion, you can ignore this section. This is useful for
matplotlib developers and packagers who may want to disable the
matplotlib version and ship a packaged version.

enthought traits 2.6
    The traits component of the Enthought Tool Suite used in the
    experimental matplotlib traits rc system. matplotlib has decided
    to stop installing this library so packagers should not distribute
    the version included with matplotlib. packagers do not need to
    list this as a requirement because the traits support is
    experimental and disabled by default.

