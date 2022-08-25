.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute. Everyone within the community
is expected to abide by our
`code of conduct <https://github.com/matplotlib/matplotlib/blob/main/CODE_OF_CONDUCT.md>`_.

The project is hosted on
https://github.com/matplotlib/matplotlib

Get Connected
=============

Do I really have something to contribute to Matplotlib?
-------------------------------------------------------

100% yes. There are so many ways to contribute to our community.

When in doubt, we recommend going together! Get connected with our community of
active contributors, many of whom felt just like you when they started out and
are happy to welcome you and support you as you get to know how we work, and
where things are. Take a look at the next sections to learn more.

Contributor incubator
---------------------

The incubator is our non-public communication channel for new contributors. It
is a private gitter room moderated by core Matplotlib developers where you can
get guidance and support for your first few PRs. It's a place you can ask
questions about anything: how to use git, github, how our PR review process
works, technical questions about the code, what makes for good documentation
or a blog post, how to get involved in community work, or get
"pre-review" on your PR.

To join, please go to our public `gitter
<https://gitter.im/matplotlib/matplotlib>`_ community channel, and ask to be
added to '#incubator'. One of our core developers will see your message and will
add you.

New Contributors meeting
------------------------

Once a month, we host a meeting to discuss topics that interest new
contributors. Anyone can attend, present, or sit in and listen to the call.
Among our attendees are fellow new contributors, as well as maintainers, and
veteran contributors, who are keen to support onboarding of new folks and
share their experience. You can find our community calendar link at the
`Scientific Python website <https://scientific-python.org/calendars/>`_, and
you can browse previous meeting notes on `github
<https://github.com/matplotlib/ProjectManagement/tree/master/
new_contributor_meeting>`_.
We recommend joining the meeting to clarify any doubts, or lingering
questions you might have, and to get to know a few of the people behind the
GitHub handles 😉. You can reach out to @noatamir on `gitter
<https://gitter.im/matplotlib/matplotlib>`_ for any clarifications or
suggestions. We <3 feedback!

.. _new_contributors:

Issues for new contributors
---------------------------

While any contributions are welcome, we have marked some issues as
particularly suited for new contributors by the label `good first issue
<https://github.com/matplotlib/matplotlib/labels/good%20first%20issue>`_. These
are well documented issues, that do not require a deep understanding of the
internals of Matplotlib. The issues may additionally be tagged with a
difficulty. ``Difficulty: Easy`` is suited for people with little Python
experience. ``Difficulty: Medium`` and ``Difficulty: Hard`` require more
programming experience. This could be for a variety of reasons, among them,
though not necessarily all at the same time:

- The issue is in areas of the code base which have more interdependencies,
  or legacy code.
- It has less clearly defined tasks, which require some independent
  exploration, making suggestions, or follow-up discussions to clarify a good
  path to resolve the issue.
- It involves Python features such as decorators and context managers, which
  have subtleties due to our implementation decisions.

In general, the Matplotlib project does not assign issues. Issues are
"assigned" or "claimed" by opening a PR; there is no other assignment
mechanism. If you have opened such a PR, please comment on the issue thread to
avoid duplication of work. Please check if there is an existing PR for the
issue you are addressing. If there is, try to work with the author by
submitting reviews of their code or commenting on the PR rather than opening
a new PR; duplicate PRs are subject to being closed.  However, if the existing
PR is an outline, unlikely to work, or stalled, and the original author is
unresponsive, feel free to open a new PR referencing the old one.

.. _submitting-a-bug-report:

Submitting a bug report
=======================

If you find a bug in the code or documentation, do not hesitate to submit a
ticket to the
`Issue Tracker <https://github.com/matplotlib/matplotlib/issues>`_. You are
also welcome to post feature requests or pull requests.

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
      '3.4.1'
      >>> import platform
      >>> platform.python_version()
      '3.9.2'

We have preloaded the issue creation page with a Markdown form that you can
use to organize this information.

Thank you for your help in keeping bug reports complete, targeted and descriptive.

.. _request-a-new-feature:

Requesting a new feature
========================

Please post feature requests to the
`Issue Tracker <https://github.com/matplotlib/matplotlib/issues>`_.

The Matplotlib developers will give feedback on the feature proposal. Since
Matplotlib is an open source project with limited resources, we encourage
users to then also
:ref:`participate in the implementation <contributing-code>`.

.. _contributing-code:

Contributing code
=================

.. _how-to-contribute:

How to contribute
-----------------

The preferred way to contribute to Matplotlib is to fork the `main
repository <https://github.com/matplotlib/matplotlib/>`__ on GitHub,
then submit a "pull request" (PR).

A brief overview is:

1. `Create an account <https://github.com/join>`_ on GitHub if you do not
   already have one.

2. Fork the `project repository <https://github.com/matplotlib/matplotlib>`_:
   click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.

3. Clone this copy to your local disk::

      git clone https://github.com/<YOUR GITHUB USERNAME>/matplotlib.git

4. Enter the directory and install the local version of Matplotlib.
   See :ref:`installing_for_devs` for instructions

5. Create a branch to hold your changes::

      git checkout -b my-feature origin/main

   and start making changes. Never work in the ``main`` branch!

6. Work on this copy, on your computer, using Git to do the version control.
   When you're done editing e.g., ``lib/matplotlib/collections.py``, do::

      git add lib/matplotlib/collections.py
      git commit

   to record your changes in Git, then push them to GitHub with::

      git push -u origin my-feature

Finally, go to the web page of your fork of the Matplotlib repo, and click
'Pull request' to send your changes to the maintainers for review.

.. seealso::

  * `Git documentation <https://git-scm.com/doc>`_
  * `Git-Contributing to a Project <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_
  * `Introduction to GitHub  <https://lab.github.com/githubtraining/introduction-to-github>`_
  * :ref:`development-workflow` for best practices for Matplotlib
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

* Formatting should follow the recommendations of PEP8_, as enforced by
  flake8_.  You can check flake8 compliance from the command line with ::

    python -m pip install flake8
    flake8 /path/to/module.py

  or your editor may provide integration with it.  Note that Matplotlib
  intentionally does not use the black_ auto-formatter (1__), in particular due
  to its unability to understand the semantics of mathematical expressions
  (2__, 3__).

  .. _PEP8: https://www.python.org/dev/peps/pep-0008/
  .. _flake8: https://flake8.pycqa.org/
  .. _black: https://black.readthedocs.io/
  .. __: https://github.com/matplotlib/matplotlib/issues/18796
  .. __: https://github.com/psf/black/issues/148
  .. __: https://github.com/psf/black/issues/1984

* Each high-level plotting function should have a simple example in the
  ``Example`` section of the docstring.  This should be as simple as possible
  to demonstrate the method.  More complex examples should go in the
  ``examples`` tree.

* Changes (both new features and bugfixes) should have good test coverage. See
  :ref:`testing` for more details.

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
  :file:`doc/api/next_api_changes/behavior`, by adding a new file with the
  naming convention ``99999-ABC.rst`` where the pull request number is followed
  by the contributor's initials. (see :file:`doc/api/api_changes.rst` for more
  information)

* See below for additional points about :ref:`keyword-argument-processing`, if
  applicable for your pull request.

.. note::

    The current state of the Matplotlib code base is not compliant with all
    of those guidelines, but we expect that enforcing those constraints on all
    new contributions will move the overall code base quality in the right
    direction.


.. seealso::

  * :ref:`coding_guidelines`
  * :ref:`testing`
  * :ref:`documenting-matplotlib`




.. _contributing_documentation:

Contributing documentation
==========================

You as an end-user of Matplotlib can make a valuable contribution because you
more clearly see the potential for improvement than a core developer. For example, you can:

- Fix a typo
- Clarify a docstring
- Write or update an :ref:`example plot <gallery>`
- Write or update a comprehensive :ref:`tutorial <tutorials>`

The documentation source files live in the same GitHub repository as the code.
Contributions are proposed and accepted through the pull request process.
For details see :ref:`how-to-contribute`.

If you have trouble getting started, you may instead open an `issue`_
describing the intended improvement.

.. _issue: https://github.com/matplotlib/matplotlib/issues

.. seealso::
  * :ref:`documenting-matplotlib`

.. _other_ways_to_contribute:

Other ways to contribute
========================

It also helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!  If Matplotlib contributes to a
project that leads to a scientific publication, please follow the
:doc:`/users/project/citing` guidelines.

.. _coding_guidelines:

Coding guidelines
=================

API changes
-----------

API consistency and stability are of great value. Therefore, API changes
(e.g. signature changes, behavior changes, removals) will only be conducted
if the added benefit is worth the user effort for adapting.

API changes in Matplotlib have to be performed following the deprecation process
below, except in very rare circumstances as deemed necessary by the development team.
This ensures that users are notified before the change will take effect and thus
prevents unexpected breaking of code.

Rules
~~~~~

- Deprecations are targeted at the next point.release (e.g. 3.x)
- Deprecated API is generally removed two point-releases after introduction
  of the deprecation. Longer deprecations can be imposed by core developers on
  a case-by-case basis to give more time for the transition
- The old API must remain fully functional during the deprecation period
- If alternatives to the deprecated API exist, they should be available
  during the deprecation period
- If in doubt, decisions about API changes are finally made by the
  API consistency lead developer

Introducing
~~~~~~~~~~~

1. Announce the deprecation in a new file
   :file:`doc/api/next_api_changes/deprecations/99999-ABC.rst` where ``99999``
   is the pull request number and ``ABC`` are the contributor's initials.
2. If possible, issue a `~matplotlib.MatplotlibDeprecationWarning` when the
   deprecated API is used. There are a number of helper tools for this:

   - Use ``_api.warn_deprecated()`` for general deprecation warnings
   - Use the decorator ``@_api.deprecated`` to deprecate classes, functions,
     methods, or properties
   - To warn on changes of the function signature, use the decorators
     ``@_api.delete_parameter``, ``@_api.rename_parameter``, and
     ``@_api.make_keyword_only``

   All these helpers take a first parameter *since*, which should be set to
   the next point release, e.g. "3.x".

   You can use standard rst cross references in *alternative*.

Expiring
~~~~~~~~

1. Announce the API changes in a new file
   :file:`doc/api/next_api_changes/[kind]/99999-ABC.rst` where ``99999``
   is the pull request number and ``ABC`` are the contributor's initials, and
   ``[kind]`` is one of the folders :file:`behavior`, :file:`development`,
   :file:`removals`. See :file:`doc/api/next_api_changes/README.rst` for more
   information. For the content, you can usually copy the deprecation notice
   and adapt it slightly.
2. Change the code functionality and remove any related deprecation warnings.

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
  in *package_data* in :file:`setupext.py`.

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

* C/C++ code in the :file:`extern/` directory is vendored, and should be kept
  close to upstream whenever possible.  It can be modified to fix bugs or
  implement new features only if the required changes cannot be made elsewhere
  in the codebase.  In particular, avoid making style fixes to it.

.. _keyword-argument-processing:

Keyword argument processing
---------------------------

Matplotlib makes extensive use of ``**kwargs`` for pass-through customizations
from one function to another.  A typical example is
`~matplotlib.axes.Axes.text`.  The definition of `matplotlib.pyplot.text` is a
simple pass-through to `matplotlib.axes.Axes.text`::

  # in pyplot.py
  def text(x, y, s, fontdict=None, **kwargs):
      return gca().text(x, y, s, fontdict=fontdict, **kwargs)

`matplotlib.axes.Axes.text` (simplified for illustration) just
passes all ``args`` and ``kwargs`` on to ``matplotlib.text.Text.__init__``::

  # in axes/_axes.py
  def text(self, x, y, s, fontdict=None, **kwargs):
      t = Text(x=x, y=y, text=s, **kwargs)

and ``matplotlib.text.Text.__init__`` (again, simplified)
just passes them on to the `matplotlib.artist.Artist.update` method::

  # in text.py
  def __init__(self, x=0, y=0, text='', **kwargs):
      super().__init__()
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

Matplotlib uses the standard Python `logging` library to write verbose
warnings, information, and debug messages. Please use it! In all those places
you write `print` calls to do your debugging, try using `logging.debug`
instead!


To include `logging` in your module, at the top of the module, you need to
``import logging``.  Then calls in your code like::

  _log = logging.getLogger(__name__)  # right after the imports

  # code
  # more code
  _log.info('Here is some information')
  _log.debug('Here is some more detailed information')

will log to a logger named ``matplotlib.yourmodulename``.

If an end-user of Matplotlib sets up `logging` to display at levels more
verbose than ``logging.WARNING`` in their code with the Matplotlib-provided
helper::

  plt.set_loglevel("debug")

or manually with ::

  import logging
  logging.basicConfig(level=logging.DEBUG)
  import matplotlib.pyplot as plt

Then they will receive messages like

.. code-block:: none

   DEBUG:matplotlib.backends:backend MacOSX version unknown
   DEBUG:matplotlib.yourmodulename:Here is some information
   DEBUG:matplotlib.yourmodulename:Here is some more detailed information

Which logging level to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are five levels at which you can emit messages.

- `logging.critical` and `logging.error` are really only there for errors that
  will end the use of the library but not kill the interpreter.
- `logging.warning` and `._api.warn_external` are used to warn the user,
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
``logging.WARNING`` to `sys.stderr`.

The `logging tutorial`_ suggests that the difference between `logging.warning`
and `._api.warn_external` (which uses `warnings.warn`) is that
`._api.warn_external` should be used for things the user must change to stop
the warning (typically in the source), whereas `logging.warning` can be more
persistent. Moreover, note that `._api.warn_external` will by default only
emit a given warning *once* for each line of user code, whereas
`logging.warning` will display the message every time it is called.

By default, `warnings.warn` displays the line of code that has the ``warn``
call. This usually isn't more informative than the warning message itself.
Therefore, Matplotlib uses `._api.warn_external` which uses `warnings.warn`,
but goes up the stack and displays the first line of code outside of
Matplotlib. For example, for the module::

    # in my_matplotlib_module.py
    import warnings

    def set_range(bottom, top):
        if bottom == top:
            warnings.warn('Attempting to set identical bottom==top')

running the script::

    from matplotlib import my_matplotlib_module
    my_matplotlib_module.set_range(0, 0)  # set range

will display

.. code-block:: none

    UserWarning: Attempting to set identical bottom==top
    warnings.warn('Attempting to set identical bottom==top')

Modifying the module to use `._api.warn_external`::

    from matplotlib import _api

    def set_range(bottom, top):
        if bottom == top:
            _api.warn_external('Attempting to set identical bottom==top')

and running the same script will display

.. code-block:: none

   UserWarning: Attempting to set identical bottom==top
   my_matplotlib_module.set_range(0, 0)  # set range

.. _logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

.. _sample-data:

Writing examples
----------------

We have hundreds of examples in subdirectories of :file:`matplotlib/examples`,
and these are automatically generated when the website is built to show up in
the :ref:`examples <gallery>` section of the website.

Any sample data that the example uses should be kept small and
distributed with Matplotlib in the
:file:`lib/matplotlib/mpl-data/sample_data/` directory.  Then in your
example code you can load it into a file handle with::

    import matplotlib.cbook as cbook
    fh = cbook.get_sample_data('mydata.dat')
