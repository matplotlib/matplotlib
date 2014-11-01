.. _troubleshooting-faq:

***************
Troubleshooting
***************

.. contents::
   :backlinks: none

.. _matplotlib-version:

Obtaining matplotlib version
============================

To find out your matplotlib version number, import it and print the
``__version__`` attribute::

    >>> import matplotlib
    >>> matplotlib.__version__
    '0.98.0'


.. _locating-matplotlib-install:

:file:`matplotlib` install location
===================================

You can find what directory matplotlib is installed in by importing it
and printing the ``__file__`` attribute::

    >>> import matplotlib
    >>> matplotlib.__file__
    '/home/jdhunter/dev/lib64/python2.5/site-packages/matplotlib/__init__.pyc'

.. _locating-matplotlib-config-dir:

:file:`.matplotlib` directory location
======================================

Each user has a matplotlib configuration directory which may contain a
:ref:`matplotlibrc <customizing-with-matplotlibrc-files>` file. To
locate your :file:`.matplotlib/` directory, use
:func:`matplotlib.get_configdir`::

    >>> import matplotlib as mpl
    >>> mpl.get_configdir()
    '/home/darren/.matplotlib'

On unix-like systems, this directory is generally located in your
:envvar:`HOME` directory.  On windows, it is in your documents and
settings directory by default::

    >>> import matplotlib
    >>> mpl.get_configdir()
    'C:\\Documents and Settings\\jdhunter\\.matplotlib'

If you would like to use a different configuration directory, you can
do so by specifying the location in your :envvar:`MPLCONFIGDIR`
environment variable -- see
:ref:`setting-linux-osx-environment-variables`.


.. _reporting-problems:

Getting help
============

There are a number of good resources for getting help with matplotlib.
There is a good chance your question has already been asked:

  - The `mailing list archive <http://matplotlib.1069221.n5.nabble.com/>`_.

  - `Github issues <https://github.com/matplotlib/matplotlib/issues>`_.

  - Stackoverflow questions tagged `matplotlib
    <http://stackoverflow.com/questions/tagged/matplotlib>`_.

If you are unable to find an answer to your question through search,
please provide the following information in your e-mail to the
`mailing list
<http://lists.sourceforge.net/mailman/listinfo/matplotlib-users>`_:

  * your operating system; (Linux/UNIX users: post the output of ``uname -a``)

  * matplotlib version::

        python -c `import matplotlib; print matplotlib.__version__`

  * where you obtained matplotlib (e.g., your Linux distribution's
    packages or the matplotlib Sourceforge site, or
    Anaconda_ or
    `Enthought Canopy <https://www.enthought.com/products/canopy/>`_).

.. _Anaconda: https://store.continuum.io/cshop/anaconda/


  * any customizations to your ``matplotlibrc`` file (see
    :ref:`customizing-matplotlib`).

  * if the problem is reproducible, please try to provide a *minimal*,
    standalone Python script that demonstrates the problem.  This is
    *the* critical step.  If you can't post a piece of code that we
    can run and reproduce your error, the chances of getting help are
    significantly diminished.  Very often, the mere act of trying to
    minimize your code to the smallest bit that produces the error
    will help you find a bug in *your* code that is causing the
    problem.

  * you can get very helpful debugging output from matlotlib by
    running your script with a ``verbose-helpful`` or
    ``--verbose-debug`` flags and posting the verbose output the
    lists::

        > python simple_plot.py --verbose-helpful > output.txt

If you compiled matplotlib yourself, please also provide

  * any changes you have made to ``setup.py`` or ``setupext.py``
  * the output of::

      rm -rf build
      python setup.py build

    The beginning of the build output contains lots of details about your
    platform that are useful for the matplotlib developers to diagnose
    your problem.

  * your compiler version -- e.g., ``gcc --version``

Including this information in your first e-mail to the mailing list
will save a lot of time.

You will likely get a faster response writing to the mailing list than
filing a bug in the bug tracker.  Most developers check the bug
tracker only periodically.  If your problem has been determined to be
a bug and can not be quickly solved, you may be asked to file a bug in
the tracker so the issue doesn't get lost.


.. _git-trouble:

Problems with recent git versions
=================================

First make sure you have a clean build and install (see
:ref:`clean-install`), get the latest git update, install it and run a
simple test script in debug mode::

    rm -rf build
    rm -rf /path/to/site-packages/matplotlib*
    git pull
    python setup.py install > build.out
    python examples/pylab_examples/simple_plot.py --verbose-debug > run.out

and post :file:`build.out` and :file:`run.out` to the
`matplotlib-devel
<http://lists.sourceforge.net/mailman/listinfo/matplotlib-devel>`_
mailing list (please do not post git problems to the `users list
<http://lists.sourceforge.net/mailman/listinfo/matplotlib-users>`_).

Of course, you will want to clearly describe your problem, what you
are expecting and what you are getting, but often a clean build and
install will help.  See also :ref:`reporting-problems`.
