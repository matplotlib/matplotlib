.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute.  We follow the `Python Software Foundation Code of Conduct
<coc_>`_ in everything we do.

The project is hosted on https://github.com/matplotlib/matplotlib

.. _coc: http://www.python.org/psf/codeofconduct/

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

When developing Matplotlib, sources must be downloaded, built, and installed
into a local environment on your machine.

We use `Git <https://git-scm.com/>`_ for version control and
`GitHub <https://github.com/>`_ for hosting our main repository.

You can check out the latest sources with the command (see
:ref:`set-up-fork` for more details)::

    git clone https://github.com/matplotlib/matplotlib.git

and navigate to the :file:`matplotlib` directory. If you have the proper privileges,
you can use ``git@`` instead of  ``https://``, which works through the ssh protocol
and might be easier to use if you are using 2-factor authentication.

Installing Matplotlib in developer mode
---------------------------------------

It is strongly recommended to set up a clean `virtual environment`_.  Do not
use on a preexisting environment!

A new environment can be set up with ::

   python3 -mvenv /path/to/devel/env

and activated with one of the following::

   source /path/to/devel/env/bin/activate  # Linux/macOS
   /path/to/devel/env/Scripts/activate.bat  # Windows cmd.exe
   /path/to/devel/env/Scripts/Activate.ps1  # Windows PowerShell

Whenever you plan to work on Matplotlib, remember to activate the development
environment in your shell!

To install Matplotlib (and compile the C-extensions) run the following
command from the top-level directory ::

   python -mpip install -ve .

This installs Matplotlib in 'editable/develop mode', i.e., builds
everything and places the correct link entries in the install
directory so that python will be able to import Matplotlib from the
source directory.  Thus, any changes to the ``*.py`` files will be
reflected the next time you import the library.  If you change the
C-extension source (which might happen if you change branches) you
will need to run ::

   python setup.py build_ext --inplace

or re-run ``python -mpip install -ve .``.

You can then run the tests to check your work environment is set up properly::

   python -mpytest

.. _virtual environment: https://docs.python.org/3/library/venv.html
.. _pytest: http://doc.pytest.org/en/latest/
.. _pep8: https://pep8.readthedocs.io/en/latest/
.. _Ghostscript: https://www.ghostscript.com/
.. _Inkscape: https://inkscape.org/

.. note::

  **Additional dependencies for testing**: pytest_ (version 3.6 or later),
  Ghostscript_, Inkscape_

.. seealso::

  * :ref:`testing`


Contributing code
=================

.. _how-to-contribute:

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
  * `Git-Contributing to a Project <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_
  * `Introduction to GitHub  <https://lab.github.com/githubtraining/introduction-to-github>`_
  * :ref:`development-workflow`
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

  In general, Matplotlib modules should **not** import `.rcParams` using ``from
  matplotlib import rcParams``, but rather access it as ``mpl.rcParams``.  This
  is because some modules are imported very early, before the `.rcParams`
  singleton is constructed.

* If your change is a major new feature, add an entry to the ``What's new``
  section by adding a new file in ``doc/users/next_whats_new`` (see
  :file:`doc/users/next_whats_new/README.rst` for more information).

* If you change the API in a backward-incompatible way, please document it in
  :file:`doc/api/api_changes`, by adding to the relevant file
  (see :file:`doc/api/api_changes.rst` for more information)

* See below for additional points about :ref:`keyword-argument-processing`, if
  applicable for your pull request.

In addition, you can check for common programming errors with the following
tools:

* Code with a good unittest coverage (at least 70%, better 100%), check with::

   python -mpip install coverage
   python -mpytest --cov=matplotlib --showlocals -v

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


.. _contributing_documentation:

Contributing documentation
==========================

Code is not the only way to contribute to Matplotlib. For instance,
documentation is also a very important part of the project and often doesn't
get as much attention as it deserves. If you find a typo in the documentation,
or have made improvements, do not hesitate to send an email to the mailing
list or submit a GitHub pull request.  To make a pull request, refer to the
guidelines outlined in :ref:`how-to-contribute`.

Full documentation can be found under the :file:`doc/`, :file:`tutorials/`,
and :file:`examples/` directories.

.. seealso::
  * :ref:`documenting-matplotlib`


.. _other_ways_to_contribute:

Other ways to contribute
=========================

It also helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!  If Matplotlib contributes to a
project that leads to a scientific publication, please follow the
:doc:`/citing` guidelines.

.. _coding_guidelines:

Coding guidelines
=================

API changes
-----------

Changes to the public API must follow a standard deprecation procedure to
prevent unexpected breaking of code that uses Matplotlib.

- Deprecations must be announced via an entry in
  the most recent :file:`doc/api/api_changes_X.Y`
- Deprecations are targeted at the next point-release (i.e. 3.x.0).
- The deprecated API should, to the maximum extent possible, remain fully
  functional during the deprecation period. In cases where this is not
  possible, the deprecation must never make a given piece of code do something
  different than it was before; at least an exception should be raised.
- If possible, usage of an deprecated API should emit a
  `.MatplotlibDeprecationWarning`. There are a number of helper tools for this:

  - Use ``cbook.warn_deprecated()`` for general deprecation warnings.
  - Use the decorator ``@cbook.deprecated`` to deprecate classes, functions,
    methods, or properties.
  - To warn on changes of the function signature, use the decorators
    ``@cbook._delete_parameter``, ``@cbook._rename_parameter``, and
    ``@cbook._make_keyword_only``.

- Deprecated API may be removed two point-releases after they were deprecated.


Adding new API
--------------

Every new function, parameter and attribute that is not explicitly marked as
private (i.e., starts with an underscore) becomes part of Matplotlib's public
API. As discussed above, changing the existing API is cumbersome. Therefore,
take particular care when adding new API:

- Mark helper functions and internal attributes as private by prefixing them
  with an underscore.
- Carefully think about good names for your functions and variables.
- Try to adopt patterns and naming conventions from existing parts of the
  Matplotlib API.
- Consider making as many arguments keyword-only as possible. See also
  `API Evolution the Right Way -- Add Parameters Compatibly`__.

  __ https://emptysqua.re/blog/api-evolution-the-right-way/#adding-parameters


New modules and files: installation
-----------------------------------

* If you have added new files or directories, or reorganized existing
  ones, make sure the new files are included in the match patterns in
  :file:`MANIFEST.in`, and/or in *package_data* in :file:`setup.py`.

C/C++ extensions
----------------

* Extensions may be written in C or C++.

* Code style should conform to PEP7 (understanding that PEP7 doesn't
  address C++, but most of its admonitions still apply).

* Python/C interface code should be kept separate from the core C/C++
  code.  The interface code should be named :file:`FOO_wrap.cpp` or
  :file:`FOO_wrapper.cpp`.

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
function, and let others pass through.  Instead of popping arguments to
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
more verbose than `logging.WARNING` in their code with the Matplotlib-provided
helper::

  plt.set_loglevel("debug")

or manually with ::

  import logging
  logging.basicConfig(level=logging.DEBUG)
  import matplotlib.pyplot as plt

Then they will receive messages like::

  DEBUG:matplotlib.backends:backend MacOSX version unknown
  DEBUG:matplotlib.yourmodulename:Here is some information
  DEBUG:matplotlib.yourmodulename:Here is some more detailed information

Which logging level to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are five levels at which you can emit messages.

- `logging.critical` and `logging.error` are really only there for errors that
  will end the use of the library but not kill the interpreter.
- `logging.warning` and `cbook._warn_external` are used to warn the user,
  see below.
- `logging.info` is for information that the user may want to know if the
  program behaves oddly. They are not displayed by default. For instance, if
  an object isn't drawn because its position is ``NaN``, that can usually
  be ignored, but a mystified user could call
  ``logging.basicConfig(level=logging.INFO)`` and get an error message that
  says why.
- `logging.debug` is the least likely to be displayed, and hence can be the
  most verbose.  "Expected" code paths (e.g., reporting normal intermediate
  steps of layouting or rendering) should only log at this level.

By default, `logging` displays all log messages at levels higher than
`logging.WARNING` to `sys.stderr`.

The `logging tutorial`_ suggests that the difference
between `logging.warning` and `cbook._warn_external` (which uses
`warnings.warn`) is that `cbook._warn_external` should be used for things the
user must change to stop the warning (typically in the source), whereas
`logging.warning` can be more persistent.  Moreover, note that
`cbook._warn_external` will by default only emit a given warning *once* for
each line of user code, whereas `logging.warning` will display the message
every time it is called.

By default, `warnings.warn` displays the line of code that has the `warn` call.
This usually isn't more informative than the warning message itself. Therefore,
Matplotlib uses `cbook._warn_external` which uses `warnings.warn`, but goes
up the stack and displays the first line of code outside of Matplotlib.
For example, for the module::

    # in my_matplotlib_module.py
    import warnings

    def set_range(bottom, top):
        if bottom == top:
            warnings.warn('Attempting to set identical bottom==top')


running the script::

    from matplotlib import my_matplotlib_module
    my_matplotlib_module.set_range(0, 0)  #set range


will display::

    UserWarning: Attempting to set identical bottom==top
    warnings.warn('Attempting to set identical bottom==top')

Modifying the module to use `cbook._warn_external`::

    from matplotlib import cbook

    def set_range(bottom, top):
        if bottom == top:
            cbook._warn_external('Attempting to set identical bottom==top')

and running the same script will display::

  UserWarning: Attempting to set identical bottom==top
  my_matplotlib_module.set_range(0, 0)  #set range

.. _logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

.. _sample-data:

Writing examples
----------------

We have hundreds of examples in subdirectories of
:file:`matplotlib/examples`, and these are automatically generated
when the website is built to show up in the `examples
<../gallery/index.html>` section of the website.

Any sample data that the example uses should be kept small and
distributed with Matplotlib in the
:file:`lib/matplotlib/mpl-data/sample_data/` directory.  Then in your
example code you can load it into a file handle with::

    import matplotlib.cbook as cbook
    fh = cbook.get_sample_data('mydata.dat')
