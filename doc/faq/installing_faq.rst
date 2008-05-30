==================
 Installation FAQ
==================

How do I report a compilation problem?
======================================

See :ref:`reporting_problems`.

How do I cleanly rebuild and reinstall everything?
==================================================

Unfortunately::

    python setup.py clean

does not properly clean the build directory, and does nothing to the
install directory.  To cleanly rebuild:

    * delete the ``build`` directory in the source tree 
    * delete ``site-packages/matplotlib`` directory in the Python
      installation.  The location of ``site-packages`` is
      platform-specific.
    * you may also want to clear some of the cache data that
      matplotlib stores in your ``.matplotlib`` directory.  You can
      find the location of this directory by doing::

          import matplotlib
          print matplotlib.get_configdir()