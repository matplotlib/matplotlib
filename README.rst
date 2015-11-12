##########
matplotlib
##########

matplotlib is a python 2D plotting library which produces publication
quality figures in a variety of hardcopy formats and interactive
environments across platforms. matplotlib can be used in python
scripts, the python and ipython shell (ala matlab or mathematica), web
application servers, and various graphical user interface toolkits.

`Home page <http://matplotlib.org/>`_

Installation
=============

For installation instructions and requirements, see the INSTALL file.

Testing
=======

After installation, you can launch the test suite::

  python tests.py

Or from the python interpreter::

  import matplotlib
  matplotlib.test()

Consider reading http://matplotlib.org/devel/coding_guide.html#testing for
more information. Note that the test suite requires nose and on python 2.7 mock
which are not installed by default. Please install with pip or your package
manager of choice.

Contact
=======
matplotlib's communication channels include active mailing lists:

* Matplotlib-announce (https://mail.python.org/mailman/listinfo/matplotlib-announce),
* Matplotlib-devel (https://mail.python.org/mailman/listinfo/matplotlib-devel),
* Matplotlib-users (https://mail.python.org/mailman/listinfo/matplotlib-users).

The latter is probably a good starting point for general questions and discussions.
