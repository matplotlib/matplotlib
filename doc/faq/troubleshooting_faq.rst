===================
Troubleshooting FAQ
===================

.. _reporting_problems:

How do I report a problem?
==========================

If you are having a problem with matplotlib, search the mailing
lists first: There's a good chance someone else has already run into
your problem.

If not, please provide the following information in your e-mail to the
`mailing list
<http://lists.sourceforge.net/mailman/listinfo/matplotlib-users>`_:

  * your operating system; on Linux/UNIX post the output of ``uname -a``
  * matplotlib version  : ``import matplotlib; print matplotlib.__version__``
  * where you obtained matplotlib (e.g. your Linux distribution's
    packages or the matplotlib Sourceforge site)
  * any customizations to your ``matplotlibrc`` file
  * if the problem is reproducible, please try to provide a *minimal*,
    standalone Python script that demonstrates the problem
  * you can get very helpful debugging output from matlotlib by
    running your script with a ``verbose-helpful`` or
    ``--verbose-debug`` flags and posting the verbose output the
    lists.

If you compiled matplotlib yourself, please also provide 

  * any changes you have made to ``setup.py`` or ``setupext.py``
  * the output of::

      rm -rf build
      python setup.py build

    The beginning of the build output contains lots of details about your
    platform that are useful for the matplotlib developers to diagnose
    your problem.  

  * your compiler version -- eg, ``gcc --version``

Including this information in your first e-mail to the mailing list
will save a lot of time.

You will likely get a faster response writing to the mailing list than
filing a bug in the bug tracker.  Most developers check the bug
tracker only periodically.  If your problem has been determined to be
a bug and can not be quickly solved, you may be asked to file a bug in
the tracker so the issue doesn't get lost.

