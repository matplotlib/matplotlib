##########
matplotlib
##########

matplotlib is a Python 2D plotting library which produces publication-quality
figures in a variety of hardcopy formats and interactive
environments across platforms. matplotlib can be used in Python
scripts, the Python and IPython shell (ala MATLAB or Mathematica), web
application servers, and various graphical user interface toolkits.

`Home page <http://matplotlib.org/>`_

Installation
=============

For installation instructions and requirements, see the INSTALL file or the `install <http://matplotlib.org/users/installing.html>`_ documentation. If you think you may want to contribute to matplotlib, check out the `guide to working with the source code <http://matplotlib.org/devel/gitwash/index.html>`_.

Testing
=======

After installation, you can launch the test suite::

  python tests.py

Or from the Python interpreter::

  import matplotlib
  matplotlib.test()

Consider reading http://matplotlib.org/devel/coding_guide.html#testing for
more information. Note that the test suite requires nose and on Python 2.7 mock
which are not installed by default. Please install with pip or your package
manager of choice.

Contact
=======
matplotlib's communication channels include active mailing lists:

* `Users <https://mail.python.org/mailman/listinfo/matplotlib-users>`_ mailing list: matplotlib-users@python.org
* `Announcement  <https://mail.python.org/mailman/listinfo/matplotlib-announce>`_ mailing list: matplotlib-announce@python.org
* `Development <https://mail.python.org/mailman/listinfo/matplotlib-devel>`_ mailing list: matplotlib-devel@python.org


The first is a good starting point for general questions and discussions.


.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/matplotlib/matplotlib
   :target: https://gitter.im/matplotlib/matplotlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Contribute
==========
You've discovered a bug or something else you want to change - excellent!

You've worked out a way to fix it – even better!

You want to tell us about it – best of all!

Start at the `contributing guide <http://matplotlib.org/devdocs/devel/contributing.html>`_!
