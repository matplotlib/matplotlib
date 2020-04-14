=================================
 MEP11: Third-party dependencies
=================================

.. contents::
   :local:

This MEP attempts to improve the way in which third-party dependencies
in matplotlib are handled.

Status
======

**Completed** -- needs to be merged

Branches and Pull requests
==========================

#1157: Use automatic dependency resolution

#1290: Debundle pyparsing

#1261: Update six to 1.2

Abstract
========

One of the goals of matplotlib has been to keep it as easy to install
as possible.  To that end, some third-party dependencies are included
in the source tree and, under certain circumstances, installed
alongside matplotlib.  This MEP aims to resolve some problems with
that approach, bring some consistency, while continuing to make
installation convenient.

At the time that was initially done, setuptools_, easy_install_ and
PyPI_ were not mature enough to be relied on.  However, at present,
we should be able to safely leverage the "modern" versions of those
tools, distribute_ and pip_.

While matplotlib has dependencies on both Python libraries and C/C++
libraries, this MEP addresses only the Python libraries so as to not
confuse the issue.  C libraries represent a larger and mostly
orthogonal set of problems.

Detailed description
====================

matplotlib depends on the following third-party Python libraries:

   - Numpy
   - dateutil (pure Python)
   - pytz (pure Python)
   - six -- required by dateutil (pure Python)
   - pyparsing (pure Python)
   - PIL (optional)
   - GUI frameworks: pygtk, gobject, tkinter, PySide, PyQt4, wx (all
     optional, but one is required for an interactive GUI)

Current behavior
----------------

When installing from source, a :program:`git` checkout or pip_:

  - :file:`setup.py` attempts to ``import numpy``.  If this fails, the
    installation fails.

  - For each of dateutil_, pytz_ and six_, :file:`setup.py` attempts to
    import them (from the top-level namespace).  If that fails,
    matplotlib installs its local copy of the library into the
    top-level namespace.

  - pyparsing_ is always installed inside of the matplotlib
    namespace.

This behavior is most surprising when used with pip_, because no
pip_ dependency resolution is performed, even though it is likely to
work for all of these packages.

The fact that pyparsing_ is installed in the matplotlib namespace has
reportedly (#1290) confused some users into thinking it is a
matplotlib-related module and import it from there rather than the
top-level.

When installing using the Windows installer, dateutil_, pytz_ and
six_ are installed at the top-level *always*, potentially overwriting
already installed copies of those libraries.

TODO: Describe behavior with the OS-X installer.

When installing using a package manager (Debian, RedHat, MacPorts
etc.), this behavior actually does the right thing, and there are no
special patches in the matplotlib packages to deal with the fact that
we handle dateutil_, pytz_ and six_ in this way.  However, care
should be taken that whatever approach we move to continues to work in
that context.

Maintaining these packages in the matplotlib tree and making sure they
are up-to-date is a maintenance burden.  Advanced new features that
may require a third-party pure Python library have a higher barrier to
inclusion because of this burden.


Desired behavior
----------------

Third-party dependencies are downloaded and installed from their
canonical locations by leveraging pip_, distribute_ and PyPI_.

dateutil_, pytz_, and pyparsing_ should be made into optional
dependencies -- though obviously some features would fail if they
aren't installed.  This will allow the user to decide whether they
want to bother installing a particular feature.

Implementation
==============

For installing from source, and assuming the user has all of the
C-level compilers and dependencies, this can be accomplished fairly
easily using distribute_ and following the instructions `here
<https://pypi.org/project/distribute>`_.  The only anticipated
change to the matplotlib library code will be to import pyparsing_
from the top-level namespace rather than from within matplotlib.  Note
that distribute_ will also allow us to remove the direct dependency
on six_, since it is, strictly speaking, only a direct dependency of
dateutil_.

For binary installations, there are a number of alternatives (here
ordered from best/hardest to worst/easiest):

    1. The distutils wininst installer allows a post-install script to
       run.  It might be possible to get this script to run pip_ to
       install the other dependencies.  (See `this thread
       <http://grokbase.com/t/python/distutils-sig/109bdnfhp4/distutils-ann-setuptools-post-install-script-for-bdist-wininst>`_
       for someone who has trod that ground before).

    2. Continue to ship dateutil_, pytz_, six_ and pyparsing_ in
       our installer, but use the post-install-script to install them
       *only* if they can not already be found.

    3. Move all of these packages inside a (new) ``matplotlib.extern``
       namespace so it is clear for outside users that these are
       external packages.  Add some conditional imports in the core
       matplotlib codebase so dateutil_ (at the top-level) is tried
       first, and failing that ``matplotlib.extern.dateutil`` is used.

2 and 3 are undesirable as they still require maintaining copies of
these packages in our tree -- and this is exacerbated by the fact that
they are used less -- only in the binary installers.  None of these 3
approaches address Numpy, which will still have to be manually
installed using an installer.

TODO: How does this relate to the Mac OS-X installer?

Backward compatibility
======================

At present, matplotlib can be installed from source on a machine
without the third party dependencies and without an internet
connection.  After this change, an internet connection (and a working
PyPI) will be required to install matplotlib for the first time.
(Subsequent matplotlib updates or development work will run without
accessing the network).

Alternatives
============

Distributing binary eggs doesn't feel like a usable solution.  That
requires getting easy_install_ installed first, and Windows users
generally prefer the well known ``.exe`` or ``.msi`` installer that works
out of the box.

.. _PyPI: https://pypi.org
.. _dateutil: https://pypi.org/project/python-dateutil/
.. _distribute: https://pypi.org/project/distribute/
.. _pip: https://pypi.org/project/pip/
.. _pyparsing: https://pypi.org/project/pyparsing/
.. _pytz: https://pypi.org/project/pytz/
.. _setuptools: https://pypi.org/project/setuptools/
.. _six: https://pypi.org/project/six/
.. _easy_install: https://setuptools.readthedocs.io/en/latest/easy_install.html
