.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute.

The project is hosted on https://github.com/matplotlib/matplotlib

Submitting a bug report
=======================

If you find a bug in the code or documentation, do not hesitate to submit a
ticket to the
`Bug Tracker <https://github.com/matplotlib/matplotlib/issues>`_. You are also
welcome to post feature requests or pull requests.

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2
   sentences.

2. A short, self-contained code snippet to reproduce the bug, ideally allowing
   a simple copy and paste to reproduce. Please do your best to reduce the code
   snippet to the minimum required.

3. The actual outcome of the code snippet.

4. The expected outcome of the code snippet.

5. The Matplotlib version, Python version and platform that you are using. You
   can grab the version with the following commands::

      >>> import matplotlib
      >>> matplotlib.__version__
      '1.5.3'
      >>> import platform
      >>> platform.python_version()
      '2.7.12'

We have preloaded the issue creation page with a Markdown template that you can
use to organize this information.

Thank you for your help in keeping bug reports complete, targeted and descriptive.

.. _installing_for_devs:

Retrieving and installing the latest version of the code
========================================================

When developing Matplotlib, sources must be downloaded, built, and installed into
a local environment on your machine.

Follow the instructions detailed :ref:`here <install_from_source>` to set up your
environment to build Matplotlib from source.

.. warning::

   When working on Matplotlib sources, having multiple versions installed by
   different methods into the same environment may not always work as expected.

To work on Matplotlib sources, it is strongly recommended to set up an alternative
development environment, using the something like `virtual environments in python
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_, or a
`conda environment <http://conda.pydata.org/docs/using/envs.html>`_.

If you choose to use an already existing environment, and not a clean virtual or
conda environment, uninstall the current version of Matplotlib in that environment
using the same method used to install it.

If working on Matplotlib documentation only, the above steps are *not* absolutely
necessary.

We use `Git <https://git-scm.com/>`_ for version control and
`GitHub <https://github.com/>`_ for hosting our main repository.

You can check out the latest sources with the command (see
:ref:`set-up-fork` for more details)::

    git clone https://github.com/matplotlib/matplotlib.git

and navigate to the :file:`matplotlib` directory. If you have the proper privileges,
you can use ``git@`` instead of  ``https://``, which works through the ssh protocol
and might be easier to use if you are using 2-factor authentication.


Building Matplotlib for image comparison tests
----------------------------------------------

Matplotlib's test suite makes heavy use of image comparison tests,
meaning the result of a plot is compared against a known good result.
Unfortunately, different versions of FreeType produce differently
formed characters, causing these image comparisons to fail.  To make
them reproducible, Matplotlib can be built with a special local copy
of FreeType.  This is recommended for all Matplotlib developers.

Copy :file:`setup.cfg.template` to :file:`setup.cfg` and edit it to contain::

  [test]
  local_freetype = True
  tests = True

or set the ``MPLLOCALFREETYPE`` environmental variable to any true
value.


Installing Matplotlib in developer mode
---------------------------------------

To install Matplotlib (and compile the c-extensions) run the following
command from the top-level directory ::

   python -mpip install -ve .

This installs Matplotlib in 'editable/develop mode', i.e., builds
everything and places the correct link entries in the install
directory so that python will be able to import Matplotlib from the
source directory.  Thus, any changes to the ``*.py`` files will be
reflected the next time you import the library.  If you change the
C-extension source (which might happen if you change branches) you
will need to run ::

   python setup.py build

or re-run ``python -mpip install -ve .``.

Alternatively, if you do ::

   python -mpip install -v .

all of the files will be copied to the installation directory however,
you will have to rerun this command every time the source is changed.
Additionally you will need to copy :file:`setup.cfg.template` to
:file:`setup.cfg` and edit it to contain ::

  [test]
  local_freetype = True
  tests = True

In either case you can then run the tests to check your work
environment is set up properly::

  pytest

.. _pytest: http://doc.pytest.org/en/latest/
.. _pep8: https://pep8.readthedocs.io/en/latest/
.. _Ghostscript: https://www.ghostscript.com/
.. _Inkscape: https://inkscape.org>

.. note::

  **Additional dependencies for testing**: pytest_ (version 3.4 or later),
  Ghostscript_, Inkscape_

.. seealso::

  * :ref:`testing`


Contributing code
=================

How to contribute
-----------------

The preferred way to contribute to Matplotlib is to fork the `main
repository <https://github.com/matplotlib/matplotlib/>`__ on GitHub,
then submit a "pull request" (PR).

The best practices for using GitHub to make PRs to Matplotlib are
documented in the :ref:`development-workflow` section.

A brief overview is:

1. `Create an account <https://github.com/join>`_ on GitHub if you do not
   already have one.

2. Fork the `project repository <https://github.com/matplotlib/matplotlib>`_:
   click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.

3. Clone this copy to your local disk::

      $ git clone https://github.com/YourLogin/matplotlib.git

4. Create a branch to hold your changes::

      $ git checkout -b my-feature origin/master

   and start making changes. Never work in the ``master`` branch!

5. Work on this copy, on your computer, using Git to do the version control.
   When you're done editing e.g., ``lib/matplotlib/collections.py``, do::

      $ git add lib/matplotlib/collections.py
      $ git commit

   to record your changes in Git, then push them to GitHub with::

      $ git push -u origin my-feature

Finally, go to the web page of your fork of the Matplotlib repo, and click
'Pull request' to send your changes to the maintainers for review.  You may
want to consider sending an email to the mailing list for more visibility.

.. seealso::

  * `Git documentation <https://git-scm.com/documentation>`_
  * :ref:`development-workflow`.
  * :ref:`using-git`

Contributing pull requests
--------------------------

It is recommended to check that your contribution complies with the following
rules before submitting a pull request:

* If your pull request addresses an issue, please use the title to describe the
  issue and mention the issue number in the pull request description to ensure
  that a link is created to the original issue.

* All public methods should have informative docstrings with sample usage when
  appropriate. Use the `numpy docstring standard
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

* Formatting should follow the recommendations of `PEP8
  <https://www.python.org/dev/peps/pep-0008/>`__. You should consider
  installing/enabling automatic PEP8 checking in your editor.  Part of the test
  suite is checking PEP8 compliance, things go smoother if the code is mostly
  PEP8 compliant to begin with.

* Each high-level plotting function should have a simple example in the
  ``Example`` section of the docstring.  This should be as simple as possible
  to demonstrate the method.  More complex examples should go in the
  ``examples`` tree.

* Changes (both new features and bugfixes) should be tested. See :ref:`testing`
  for more details.

* Import the following modules using the standard scipy conventions::

     import numpy as np
     import numpy.ma as ma
     import matplotlib as mpl
     import matplotlib.pyplot as plt
     import matplotlib.cbook as cbook
     import matplotlib.patches as mpatches

* If your change is a major new feature, add an entry to the ``What's new``
  section by adding a new file in ``doc/users/next_whats_new`` (see
  :file:`doc/users/next_whats_new/README.rst` for more information).

* If you change the API in a backward-incompatible way, please document it in
  `doc/api/api_changes`, by adding a new file describing your changes (see
  :file:`doc/api/api_changes/README.rst` for more information)

* See below for additional points about :ref:`keyword-argument-processing`, if
  applicable for your pull request.

In addition, you can check for common programming errors with the following
tools:

* Code with a good unittest coverage (at least 70%, better 100%), check with::

   python -mpip install coverage
   pytest --cov=matplotlib --showlocals -v

* No pyflakes warnings, check with::

   python -mpip install pyflakes
   pyflakes path/to/module.py

.. note::

    The current state of the Matplotlib code base is not compliant with all
    of those guidelines, but we expect that enforcing those constraints on all
    new contributions will move the overall code base quality in the right
    direction.


.. seealso::

  * :ref:`coding_guidelines`
  * :ref:`testing`
  * :ref:`documenting-matplotlib`



.. _new_contributors:

Issues for New Contributors
---------------------------

New contributors should look for the following tags when looking for issues.
We strongly recommend that new contributors tackle issues labeled
`good first issue <https://github.com/matplotlib/matplotlib/labels/good%20first%20issue>`_
as they are easy, well documented issues, that do not require an understanding of
the different submodules of Matplotlib.
This helps the contributor become familiar with the contribution
workflow, and for the core devs to become acquainted with the contributor;
besides which, we frequently underestimate how easy an issue is to solve!

.. _other_ways_to_contribute:

Other ways to contribute
=========================


Code is not the only way to contribute to Matplotlib. For instance,
documentation is also a very important part of the project and often doesn't
get as much attention as it deserves. If you find a typo in the documentation,
or have made improvements, do not hesitate to send an email to the mailing
list or submit a GitHub pull request. Full documentation can be found under
the doc/ directory.

It also helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!

.. _coding_guidelines:

Coding guidelines
=================

New modules and files: installation
-----------------------------------

* If you have added new files or directories, or reorganized existing
  ones, make sure the new files are included in the match patterns in
  :file:`MANIFEST.in`, and/or in `package_data` in `setup.py`.

C/C++ extensions
----------------

* Extensions may be written in C or C++.

* Code style should conform to PEP7 (understanding that PEP7 doesn't
  address C++, but most of its admonitions still apply).

* Python/C interface code should be kept separate from the core C/C++
  code.  The interface code should be named `FOO_wrap.cpp` or
  `FOO_wrapper.cpp`.

* Header file documentation (aka docstrings) should be in Numpydoc
  format.  We don't plan on using automated tools for these
  docstrings, and the Numpydoc format is well understood in the
  scientific Python community.

.. _keyword-argument-processing:

Keyword argument processing
---------------------------

Matplotlib makes extensive use of ``**kwargs`` for pass-through
customizations from one function to another.  A typical example is in
:func:`matplotlib.pyplot.text`.  The definition of the pylab text
function is a simple pass-through to
:meth:`matplotlib.axes.Axes.text`::

  # in pylab.py
  def text(*args, **kwargs):
      ret =  gca().text(*args, **kwargs)
      draw_if_interactive()
      return ret

:meth:`~matplotlib.axes.Axes.text` in simplified form looks like this,
i.e., it just passes all ``args`` and ``kwargs`` on to
:meth:`matplotlib.text.Text.__init__`::

  # in axes/_axes.py
  def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
      t = Text(x=x, y=y, text=s, **kwargs)

and :meth:`~matplotlib.text.Text.__init__` (again with liberties for
illustration) just passes them on to the
:meth:`matplotlib.artist.Artist.update` method::

  # in text.py
  def __init__(self, x=0, y=0, text='', **kwargs):
      Artist.__init__(self)
      self.update(kwargs)

``update`` does the work looking for methods named like
``set_property`` if ``property`` is a keyword argument.  i.e., no one
looks at the keywords, they just get passed through the API to the
artist constructor which looks for suitably named methods and calls
them with the value.

As a general rule, the use of ``**kwargs`` should be reserved for
pass-through keyword arguments, as in the example above.  If all the
keyword args are to be used in the function, and not passed
on, use the key/value keyword args in the function definition rather
than the ``**kwargs`` idiom.

In some cases, you may want to consume some keys in the local
function, and let others pass through.  Instead of poping arguments to
use off ``**kwargs``, specify them as keyword-only arguments to the local
function.  This makes it obvious at a glance which arguments will be
consumed in the function.  For example, in
:meth:`~matplotlib.axes.Axes.plot`, ``scalex`` and ``scaley`` are
local arguments and the rest are passed on as
:meth:`~matplotlib.lines.Line2D` keyword arguments::

  # in axes/_axes.py
  def plot(self, *args, scalex=True, scaley=True, **kwargs):
      lines = []
      for line in self._get_lines(*args, **kwargs):
          self.add_line(line)
          lines.append(line)

.. _using_logging:

Using logging for debug messages
--------------------------------

Matplotlib uses the standard python `logging` library to write verbose
warnings, information, and
debug messages.  Please use it!  In all those places you write :func:`print()`
statements to do your debugging, try using :func:`log.debug()` instead!


To include `logging` in your module, at the top of the module, you need to
``import logging``.  Then calls in your code like::

  _log = logging.getLogger(__name__)  # right after the imports

  # code
  # more code
  _log.info('Here is some information')
  _log.debug('Here is some more detailed information')

will log to a logger named ``matplotlib.yourmodulename``.

If an end-user of Matplotlib sets up `logging` to display at levels
more verbose than `logger.WARNING` in their code as follows::

  import logging
  fmt = '%(name)s:%(lineno)5d - %(levelname)s - %(message)s'
  logging.basicConfig(level=logging.DEBUG, format=fmt)
  import matplotlib.pyplot as plt

Then they will receive messages like::

  matplotlib.backends:   89 - INFO - backend MacOSX version unknown
  matplotlib.yourmodulename: 347 - INFO - Here is some information
  matplotlib.yourmodulename: 348 - DEBUG - Here is some more detailed information

Which logging level to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are five levels at which you can emit messages.
`logging.critical` and `logging.error`
are really only there for errors that will end the use of the library but
not kill the interpreter.  `logging.warning` overlaps with the
``warnings`` library.  The
`logging tutorial <https://docs.python.org/3/howto/logging.html#logging-basic-tutorial>`_
suggests that the difference
between `logging.warning` and `warnings.warn` is that
`warnings.warn` be used for things the user must change to stop
the warning, whereas `logging.warning` can be more persistent.

By default, `logging` displays all log messages at levels higher than
`logging.WARNING` to `sys.stderr`.

Calls to `logging.info` are not displayed by default.  They are for
information that the user may want to know if the program behaves oddly.
For instance, if an object isn't drawn because its position is ``NaN``,
that can usually be ignored, but a mystified user could set
``logging.basicConfig(level=logging.INFO)`` and get an error message that
says why.

`logging.debug` is the least likely to be displayed, and hence can
be the most verbose.

.. _custom_backend:

Developing a new backend
------------------------

If you are working on a custom backend, the *backend* setting in
:file:`matplotlibrc` (:doc:`/tutorials/introductory/customizing`) supports an
external backend via the ``module`` directive.  If
:file:`my_backend.py` is a Matplotlib backend in your
:envvar:`PYTHONPATH`, you can set it on one of several ways

* in :file:`matplotlibrc`::

    backend : module://my_backend

* with the :envvar:`MPLBACKEND` environment variable::

    > export MPLBACKEND="module://my_backend"
    > python simple_plot.py

* with the use directive in your script::

    import matplotlib
    matplotlib.use('module://my_backend')

.. _sample-data:

Writing examples
----------------

We have hundreds of examples in subdirectories of
:file:`matplotlib/examples`, and these are automatically generated
when the website is built to show up in the `examples
<../gallery/index.html>` section of the website.

Any sample data that the example uses should be kept small and
distributed with Matplotlib in the
`lib/matplotlib/mpl-data/sample_data/` directory.  Then in your
example code you can load it into a file handle with::

    import matplotlib.cbook as cbook
    fh = cbook.get_sample_data('mydata.dat')
