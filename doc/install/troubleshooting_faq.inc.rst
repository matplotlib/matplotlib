.. _troubleshooting-install:

.. redirect-from:: /users/installing/troubleshooting_faq

Troubleshooting
===============

.. _matplotlib-version:

Obtaining Matplotlib version
----------------------------

To find out your Matplotlib version number, import it and print the
``__version__`` attribute::

    >>> import matplotlib
    >>> matplotlib.__version__
    '0.98.0'


.. _locating-matplotlib-install:

:file:`matplotlib` install location
-----------------------------------

You can find what directory Matplotlib is installed in by importing it
and printing the ``__file__`` attribute::

    >>> import matplotlib
    >>> matplotlib.__file__
    '/home/jdhunter/dev/lib64/python2.5/site-packages/matplotlib/__init__.pyc'


.. _locating-matplotlib-config-dir:

:file:`matplotlib` configuration and cache directory locations
--------------------------------------------------------------

Each user has a Matplotlib configuration directory which may contain a
:ref:`matplotlibrc <customizing-with-matplotlibrc-files>` file. To
locate your :file:`matplotlib/` configuration directory, use
:func:`matplotlib.get_configdir`::

    >>> import matplotlib as mpl
    >>> mpl.get_configdir()
    '/home/darren/.config/matplotlib'

On Unix-like systems, this directory is generally located in your
:envvar:`HOME` directory under the :file:`.config/` directory.

In addition, users have a cache directory. On Unix-like systems, this is
separate from the configuration directory by default. To locate your
:file:`.cache/` directory, use :func:`matplotlib.get_cachedir`::

    >>> import matplotlib as mpl
    >>> mpl.get_cachedir()
    '/home/darren/.cache/matplotlib'

On Windows, both the config directory and the cache directory are
the same and are in your :file:`Documents and Settings` or :file:`Users`
directory by default::

    >>> import matplotlib as mpl
    >>> mpl.get_configdir()
    'C:\\Documents and Settings\\jdhunter\\.matplotlib'
    >>> mpl.get_cachedir()
    'C:\\Documents and Settings\\jdhunter\\.matplotlib'

If you would like to use a different configuration directory, you can
do so by specifying the location in your :envvar:`MPLCONFIGDIR`
environment variable -- see
:ref:`setting-linux-macos-environment-variables`.  Note that
:envvar:`MPLCONFIGDIR` sets the location of both the configuration
directory and the cache directory.
