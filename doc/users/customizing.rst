.. _customizing-matplotlib:

**********************
Customizing matplotlib
**********************

.. _customizing-with-matplotlibrc-files:

The :file:`matplotlibrc` file
=============================

matplotlib uses :file:`matplotlibrc` configuration files to customize all kinds
of properties, which we call `rc settings` or `rc parameters`. You can control
the defaults of almost every property in matplotlib: figure size and dpi, line
width, color and style, axes, axis and grid properties, text and font
properties and so on. matplotlib looks for :file:`matplotlibrc` in three
locations, in the following order:

1. :file:`matplotlibrc` in the current working directory, usually used for
   specific customizations that you do not want to apply elsewhere.

2. It next looks in a user-specific place, depending on your platform:

   - On Linux, it looks in :file:`.config/matplotlib/matplotlibrc` (or
     `$XDG_CONFIG_HOME/matplotlib/matplotlibrc`) if you've customized
     your environment.

   - On other platforms, it looks in :file:`.matplotlib/matplotlibrc`.

   See :ref:`locating-matplotlib-config-dir`.

3. :file:`{INSTALL}/matplotlib/mpl-data/matplotlibrc`, where
   :file:`{INSTALL}` is something like
   :file:`/usr/lib/python2.5/site-packages` on Linux, and maybe
   :file:`C:\\Python25\\Lib\\site-packages` on Windows. Every time you
   install matplotlib, this file will be overwritten, so if you want
   your customizations to be saved, please move this file to your
   user-specific matplotlib directory.

To display where the currently active :file:`matplotlibrc` file was
loaded from, one can do the following::

  >>> import matplotlib
  >>> matplotlib.matplotlib_fname()
  '/home/foo/.config/matplotlib/matplotlibrc'

See below for a sample :ref:`matplotlibrc file<matplotlibrc-sample>`.

.. _customizing-with-dynamic-rc-settings:

Dynamic rc settings
===================

You can also dynamically change the default rc settings in a python script or
interactively from the python shell. All of the rc settings are stored in a
dictionary-like variable called :data:`matplotlib.rcParams`, which is global to
the matplotlib package. rcParams can be modified directly, for example::

    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.color'] = 'r'

Matplotlib also provides a couple of convenience functions for modifying rc
settings. The :func:`matplotlib.rc` command can be used to modify multiple
settings in a single group at once, using keyword arguments::

    import matplotlib as mpl
    mpl.rc('lines', linewidth=2, color='r')

The :func:`matplotlib.rcdefaults` command will restore the standard matplotlib
default settings.

There is some degree of validation when setting the values of rcParams, see
:mod:`matplotlib.rcsetup` for details.


.. _matplotlibrc-sample:

A sample matplotlibrc file
--------------------------------------------------------------------

.. htmlonly::

    `(download) <../_static/matplotlibrc>`__

.. literalinclude:: ../../lib/matplotlib/mpl-data/matplotlibrc
