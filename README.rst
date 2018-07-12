|Travis|_ |AppVeyor|_ |Codecov|_ |PyPi|_ |Gitter|_ |NUMFocus|_


.. |Travis| image:: https://travis-ci.org/matplotlib/matplotlib.svg?branch=master
.. _Travis: https://travis-ci.org/matplotlib/matplotlib

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/github/matplotlib/matplotlib?branch=master&svg=true
.. _AppVeyor: https://ci.appveyor.com/project/matplotlib/matplotlib

.. |Codecov| image:: https://codecov.io/github/matplotlib/matplotlib/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/matplotlib/matplotlib?branch=master

.. |PyPi| image:: https://badge.fury.io/py/matplotlib.svg
.. _PyPi: https://badge.fury.io/py/matplotlib

.. |Gitter| image:: https://badges.gitter.im/matplotlib/matplotlib.png
.. _Gitter: https://gitter.im/matplotlib/matplotlib

.. |NUMFocus| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
.. _NUMFocus: http://www.numfocus.org


##########
Matplotlib
##########

Matplotlib is a Python 2D plotting library which produces publication-quality
figures in a variety of hardcopy formats and interactive environments across
platforms. Matplotlib can be used in Python scripts, the Python and IPython
shell (à la MATLAB or Mathematica), web application servers, and various
graphical user interface toolkits.

`Home page <http://matplotlib.org/>`_

Installation
============

For installation instructions and requirements, see the INSTALL.rst file or the
`install <http://matplotlib.org/users/installing.html>`_ documentation. If you
think you may want to contribute to matplotlib, check out the `guide to
working with the source code
<http://matplotlib.org/devel/gitwash/index.html>`_.

Testing
=======

After installation, you can launch the test suite::

  py.test

Or from the Python interpreter::

  import matplotlib
  matplotlib.test()

Consider reading http://matplotlib.org/devel/coding_guide.html#testing for
more information. Note that the test suite requires pytest and, on Python 2.7,
mock. Please install with pip or your package manager of choice.

Contact
=======
matplotlib's communication channels include active mailing lists:

* `Users <https://mail.python.org/mailman/listinfo/matplotlib-users>`_ mailing list: matplotlib-users@python.org
* `Announcement  <https://mail.python.org/mailman/listinfo/matplotlib-announce>`_ mailing list: matplotlib-announce@python.org
* `Development <https://mail.python.org/mailman/listinfo/matplotlib-devel>`_ mailing list: matplotlib-devel@python.org

The first is a good starting point for general questions and discussions.

Gitter_ is for coordinating development and asking questions directly related
to contributing to matplotlib.

Contribute
==========
You've discovered a bug or something else you want to change - excellent!

You've worked out a way to fix it – even better!

You want to tell us about it – best of all!

Start at the `contributing guide <http://matplotlib.org/devdocs/devel/contributing.html>`_!

Developer notes are now at `_Developer Discussions <https://github.com/orgs/matplotlib/teams/developers/discussions>`_
