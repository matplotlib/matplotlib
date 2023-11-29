.. redirect-from:: /devel/contributing

.. _contributing:

==========
Contribute
==========

You've discovered a bug or something else you want to change
in Matplotlib — excellent!

You've worked out a way to fix it — even better!

You want to tell us about it — best of all!

This project is a community effort, and everyone is welcome to contribute. Everyone
within the community is expected to abide by our `code of conduct
<https://github.com/matplotlib/matplotlib/blob/main/CODE_OF_CONDUCT.md>`_.

Below, you can find a number of ways to contribute, and how to connect with the
Matplotlib community.

.. _start-contributing:

Get started
===========

There is no pre-defined pathway for new contributors -- we recommend looking at
existing issue and pull request discussions, and following the conversations
during pull request reviews to get context. Or you can deep-dive into a subset
of the code-base to understand what is going on.

There are a few typical new contributor profiles:

* **You are a Matplotlib user, and you see a bug, a potential improvement, or
  something that annoys you, and you can fix it.**

  You can search our issue tracker for an existing issue that describes your problem or
  open a new issue to inform us of the problem you observed and discuss the best approach
  to fix it. If your contributions would not be captured on GitHub (social media,
  communication, educational content), you can also reach out to us on gitter_,
  `Discourse <https://discourse.matplotlib.org/>`__ or attend any of our `community
  meetings <https://scientific-python.org/calendars>`__.

* **You are not a regular Matplotlib user but a domain expert: you know about
  visualization, 3D plotting, design, technical writing, statistics, or some
  other field where Matplotlib could be improved.**

  Awesome -- you have a focus on a specific application and domain and can
  start there. In this case, maintainers can help you figure out the best
  implementation; open an issue or pull request with a starting point, and we'll
  be happy to discuss technical approaches.

  If you prefer, you can use the `GitHub functionality for "draft" pull requests
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request#converting-a-pull-request-to-a-draft>`__
  and request early feedback on whatever you are working on, but you should be
  aware that maintainers may not review your contribution unless it has the
  "Ready to review" state on GitHub.

* **You are new to Matplotlib, both as a user and contributor, and want to start
  contributing but have yet to develop a particular interest.**

  Having some previous experience or relationship with the library can be very
  helpful when making open-source contributions. It helps you understand why
  things are the way they are and how they *should* be. Having first-hand
  experience and context is valuable both for what you can bring to the
  conversation (and given the breadth of Matplotlib's usage, there is a good
  chance it is a unique context in any given conversation) and make it easier to
  understand where other people are coming from.

  Understanding the entire codebase is a long-term project, and nobody expects
  you to do this right away. If you are determined to get started with
  Matplotlib and want to learn, going through the basic functionality,
  choosing something to focus on (3d, testing, documentation, animations, etc.)
  and gaining context on this area by reading the issues and pull requests
  touching these subjects is a reasonable approach.

.. _get_connected:

Get connected
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
is a private gitter_ (chat) room moderated by core Matplotlib developers where
you can get guidance and support for your first few PRs. It's a place where you
can ask questions about anything: how to use git, GitHub, how our PR review
process works, technical questions about the code, what makes for good
documentation or a blog post, how to get involved in community work, or get a
"pre-review" on your PR.

To join, please go to our public community_ channel, and ask to be added to
``#incubator``. One of our core developers will see your message and will add you.

New Contributors Meeting
------------------------

Once a month, we host a meeting to discuss topics that interest new
contributors. Anyone can attend, present, or sit in and listen to the call.
Among our attendees are fellow new contributors, as well as maintainers, and
veteran contributors, who are keen to support onboarding of new folks and
share their experience. You can find our community calendar link at the
`Scientific Python website <https://scientific-python.org/calendars/>`_, and
you can browse previous meeting notes on `GitHub
<https://github.com/matplotlib/ProjectManagement/tree/master/new_contributor_meeting>`_.
We recommend joining the meeting to clarify any doubts, or lingering
questions you might have, and to get to know a few of the people behind the
GitHub handles 😉. You can reach out to us on gitter_ for any clarifications or
suggestions. We ❤ feedback!

.. _new_contributors:

Good first issues
-----------------

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

.. _managing_issues_prs:

Work on an issue
----------------

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

Submit a bug report
===================

If you find a bug in the code or documentation, do not hesitate to submit a
ticket to the
`Issue Tracker <https://github.com/matplotlib/matplotlib/issues>`_. You are
also welcome to post feature requests or pull requests.

If you are reporting a bug, please do your best to include the following:

#. A short, top-level summary of the bug. In most cases, this should be 1-2
   sentences.

#. A short, self-contained code snippet to reproduce the bug, ideally allowing
   a simple copy and paste to reproduce. Please do your best to reduce the code
   snippet to the minimum required.

#. The actual outcome of the code snippet.

#. The expected outcome of the code snippet.

#. The Matplotlib version, Python version and platform that you are using. You
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

Request a new feature
=====================

Please post feature requests to the
`Issue Tracker <https://github.com/matplotlib/matplotlib/issues>`_.

The Matplotlib developers will give feedback on the feature proposal. Since
Matplotlib is an open source project with limited resources, we encourage
users to then also
:ref:`participate in the implementation <contributing-code>`.

.. _contributing-code:

Contribute code
===============

.. _how-to-contribute:

How to contribute
-----------------

The preferred way to contribute to Matplotlib is to fork the `main
repository <https://github.com/matplotlib/matplotlib/>`__ on GitHub,
then submit a "pull request" (PR). You can do this by cloning a copy of the
Maplotlib repository to your own computer, or alternatively using
`GitHub Codespaces <https://docs.github.com/codespaces>`_, a cloud-based
in-browser development environment that comes with the appropriated setup to
contribute to Matplotlib.

Workflow overview
^^^^^^^^^^^^^^^^^

A brief overview of the workflow is as follows.

#. `Create an account <https://github.com/join>`_ on GitHub if you do not
   already have one.

#. Fork the `project repository <https://github.com/matplotlib/matplotlib>`_ by
   clicking on the :octicon:`repo-forked` **Fork** button near the top of the page.
   This creates a copy of the code under your account on the GitHub server.

#. Set up a development environment:

   .. tab-set::

      .. tab-item:: Local development

          Clone this copy to your local disk::

            git clone https://github.com/<YOUR GITHUB USERNAME>/matplotlib.git

      .. tab-item:: Using GitHub Codespaces

          Check out the Matplotlib repository and activate your development environment:

          #. Open codespaces on your fork by clicking on the green "Code" button
             on the GitHub web interface and selecting the "Codespaces" tab.

          #. Next, click on "Open codespaces on <your branch name>". You will be
             able to change branches later, so you can select the default
             ``main`` branch.

          #. After the codespace is created, you will be taken to a new browser
             tab where you can use the terminal to activate a pre-defined conda
             environment called ``mpl-dev``::

              conda activate mpl-dev


#. Install the local version of Matplotlib with::

     python -m pip install --no-build-isolation --editable .[dev]

   See :ref:`installing_for_devs` for detailed instructions.

#. Create a branch to hold your changes::

     git checkout -b my-feature origin/main

   and start making changes. Never work in the ``main`` branch!

#. Work on this task using Git to do the version control. Codespaces persist for
   some time (check the `documentation for details
   <https://docs.github.com/codespaces/getting-started/the-codespace-lifecycle>`_)
   and can be managed on https://github.com/codespaces. When you're done editing
   e.g., ``lib/matplotlib/collections.py``, do::

     git add lib/matplotlib/collections.py
     git commit

   to record your changes in Git, then push them to your GitHub fork with::

     git push -u origin my-feature

Open a pull request on Matplotlib
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, go to the web page of *your fork* of the Matplotlib repo, and click
**Compare & pull request** to send your changes to the maintainers for review.
The base repository is ``matplotlib/matplotlib`` and the base branch is
generally ``main``. For more guidance, see GitHub's `pull request tutorial
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.

For more detailed instructions on how to set up Matplotlib for development and
best practices for contribution, see :ref:`installing_for_devs`.

GitHub Codespaces workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If you need to open a GUI window with Matplotlib output on Codespaces, our
  configuration includes a `light-weight Fluxbox-based desktop
  <https://github.com/devcontainers/features/tree/main/src/desktop-lite>`_.
  You can use it by connecting to this desktop via your web browser. To do this:

  #. Press ``F1`` or ``Ctrl/Cmd+Shift+P`` and select
     ``Ports: Focus on Ports View`` in the VSCode session to bring it into
     focus. Open the ports view in your tool, select the ``noVNC`` port, and
     click the Globe icon.
  #. In the browser that appears, click the Connect button and enter the desktop
     password (``vscode`` by default).

  Check the `GitHub instructions
  <https://github.com/devcontainers/features/tree/main/src/desktop-lite#connecting-to-the-desktop>`_
  for more details on connecting to the desktop.

* If you also built the documentation pages, you can view them using Codespaces.
  Use the "Extensions" icon in the activity bar to install the "Live Server"
  extension. Locate the ``doc/build/html`` folder in the Explorer, right click
  the file you want to open and select "Open with Live Server."

.. _contributing_documentation:

Contribute documentation
========================

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

While the current state of the Matplotlib code base is not compliant with all
of these guidelines, our goal in enforcing these constraints on new
contributions is that it improves the readability and consistency of the code base
going forward.

PEP8, as enforced by flake8
---------------------------

Formatting should follow the recommendations of PEP8_, as enforced by flake8_.
Matplotlib modifies PEP8 to extend the maximum line length to 88
characters. You can check flake8 compliance from the command line with ::

    python -m pip install flake8
    flake8 /path/to/module.py

or your editor may provide integration with it.  Note that Matplotlib intentionally
does not use the black_ auto-formatter (1__), in particular due to its inability
to understand the semantics of mathematical expressions (2__, 3__).

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _flake8: https://flake8.pycqa.org/
.. _black: https://black.readthedocs.io/
.. __: https://github.com/matplotlib/matplotlib/issues/18796
.. __: https://github.com/psf/black/issues/148
.. __: https://github.com/psf/black/issues/1984


Package imports
---------------
Import the following modules using the standard scipy conventions::

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

Variable names
--------------

When feasible, please use our internal variable naming convention for objects
of a given class and objects of any child class:

+------------------------------------+---------------+------------------------------------------+
|             base class             | variable      |                multiples                 |
+====================================+===============+==========================================+
| `~matplotlib.figure.FigureBase`    | ``fig``       |                                          |
+------------------------------------+---------------+------------------------------------------+
| `~matplotlib.axes.Axes`            | ``ax``        |                                          |
+------------------------------------+---------------+------------------------------------------+
| `~matplotlib.transforms.Transform` | ``trans``     | ``trans_<source>_<target>``              |
+                                    +               +                                          +
|                                    |               | ``trans_<source>`` when target is screen |
+------------------------------------+---------------+------------------------------------------+

Generally, denote more than one instance of the same class by adding suffixes to
the variable names. If a format isn't specified in the table, use numbers or
letters as appropriate.


.. _type-hints:

Type hints
----------

If you add new public API or change public API, update or add the
corresponding `mypy <https://mypy.readthedocs.io/en/latest/>`_ type hints.
We generally use `stub files
<https://typing.readthedocs.io/en/latest/source/stubs.html#type-stubs>`_
(``*.pyi``) to store the type information; for example ``colors.pyi`` contains
the type information for ``colors.py``. A notable exception is ``pyplot.py``,
which is type hinted inline.

Type hints are checked by the mypy :ref:`pre-commit hook <pre-commit-hooks>`
and can often be verified using ``tools\stubtest.py`` and occasionally may
require the use of ``tools\check_typehints.py``.


.. _new-changed-api:

API changes and new features
----------------------------

API consistency and stability are of great value; Therefore, API changes
(e.g. signature changes, behavior changes, removals) will only be conducted
if the added benefit is worth the effort of adapting existing code.

Because we are a visualization library, our primary output is the final
visualization the user sees; therefore, the appearance of the figure is part of
the API and any changes, either semantic or :ref:`esthetic <color_changes>`,
are backwards-incompatible API changes.

.. _api_whats_new:

Announce changes, deprecations, and new features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When adding or changing the API in a backward in-compatible way, please add the
appropriate :ref:`versioning directive <versioning-directives>` and document it
for the release notes and add the entry to the appropriate folder:

+-------------------+-----------------------------+----------------------------------------------+
| addition          |   versioning directive      |  announcement folder                         |
+===================+=============================+==============================================+
| new feature       | ``.. versionadded:: 3.N``   | :file:`doc/users/next_whats_new/`            |
+-------------------+-----------------------------+----------------------------------------------+
| API change        | ``.. versionchanged:: 3.N`` | :file:`doc/api/next_api_changes/[kind]`      |
+-------------------+-----------------------------+----------------------------------------------+

API deprecations are first introduced and then expired. During the introduction
period, users are warned that the API *will* change in the future.
During the expiration period, code is changed as described in the notice posted
during the introductory period.

+-----------+--------------------------------------------------+----------------------------------------------+
|   stage   |                 required changes                 |             announcement folder              |
+===========+==================================================+==============================================+
| introduce | :ref:`introduce deprecation <intro-deprecation>` | :file:`doc/api/next_api_changes/deprecation` |
+-----------+--------------------------------------------------+----------------------------------------------+
| expire    | :ref:`expire deprecation <expire-deprecation>`   | :file:`doc/api/next_api_changes/[kind]`      |
+-----------+--------------------------------------------------+----------------------------------------------+

For both change notes and what's new, please avoid using references in section
titles, as it causes links to be confusing in the table of contents. Instead,
ensure that a reference is included in the descriptive text.

API Change Notes
""""""""""""""""
.. include:: ../api/next_api_changes/README.rst
   :start-line: 5
   :end-line: 31

What's new
""""""""""
.. include:: ../users/next_whats_new/README.rst
   :start-line: 5
   :end-line: 24


Deprecation
^^^^^^^^^^^
API changes in Matplotlib have to be performed following the deprecation process
below, except in very rare circumstances as deemed necessary by the development
team. This ensures that users are notified before the change will take effect
and thus prevents unexpected breaking of code.

Rules
"""""
- Deprecations are targeted at the next point.release (e.g. 3.x)
- Deprecated API is generally removed two point-releases after introduction
  of the deprecation. Longer deprecations can be imposed by core developers on
  a case-by-case basis to give more time for the transition
- The old API must remain fully functional during the deprecation period
- If alternatives to the deprecated API exist, they should be available
  during the deprecation period
- If in doubt, decisions about API changes are finally made by the
  API consistency lead developer

.. _intro-deprecation:

Introduce deprecation
"""""""""""""""""""""

#. Create :ref:`deprecation notice <api_whats_new>`

#. If possible, issue a `~matplotlib.MatplotlibDeprecationWarning` when the
   deprecated API is used. There are a number of helper tools for this:

   - Use ``_api.warn_deprecated()`` for general deprecation warnings
   - Use the decorator ``@_api.deprecated`` to deprecate classes, functions,
     methods, or properties
   - Use ``@_api.deprecate_privatize_attribute`` to annotate deprecation of
     attributes while keeping the internal private version.
   - To warn on changes of the function signature, use the decorators
     ``@_api.delete_parameter``, ``@_api.rename_parameter``, and
     ``@_api.make_keyword_only``

   All these helpers take a first parameter *since*, which should be set to
   the next point release, e.g. "3.x".

   You can use standard rst cross references in *alternative*.

#. Make appropriate changes to the type hints in the associated ``.pyi`` file.
   The general guideline is to match runtime reported behavior.

   - Items marked with ``@_api.deprecated`` or ``@_api.deprecate_privatize_attribute``
     are generally kept during the expiry period, and thus no changes are needed on
     introduction.
   - Items decorated with ``@_api.rename_parameter`` or ``@_api.make_keyword_only``
     report the *new* (post deprecation) signature at runtime, and thus *should* be
     updated on introduction.
   - Items decorated with ``@_api.delete_parameter`` should include a default value hint
     for the deleted parameter, even if it did not previously have one (e.g.
     ``param: <type> = ...``).

.. _expire-deprecation:

Expire deprecation
""""""""""""""""""

#. Create :ref:`deprecation announcement <api_whats_new>`. For the content,
   you can usually copy the deprecation notice and adapt it slightly.

#. Change the code functionality and remove any related deprecation warnings.

#. Make appropriate changes to the type hints in the associated ``.pyi`` file.

   - Items marked with ``@_api.deprecated`` or ``@_api.deprecate_privatize_attribute``
     are to be removed on expiry.
   - Items decorated with ``@_api.rename_parameter`` or ``@_api.make_keyword_only``
     will have been updated at introduction, and require no change now.
   - Items decorated with ``@_api.delete_parameter`` will need to be updated to the
     final signature, in the same way as the ``.py`` file signature is updated.
   - Any entries in :file:`ci/mypy-stubtest-allowlist.txt` which indicate a deprecation
     version should be double checked. In most cases this is not needed, though some
     items were never type hinted in the first place and were added to this file
     instead. For removed items that were not in the stub file, only deleting from the
     allowlist is required.

Adding new API and features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


.. _versioning-directives:

Versioning directives
"""""""""""""""""""""

When making a backward incompatible change, please add a versioning directive in
the docstring. The directives should be placed at the end of a description block.
For example::

  class Foo:
      """
      This is the summary.

      Followed by a longer description block.

      Consisting of multiple lines and paragraphs.

      .. versionadded:: 3.5

      Parameters
      ----------
      a : int
          The first parameter.
      b: bool, default: False
          This was added later.

          .. versionadded:: 3.6
      """

      def set_b(b):
          """
          Set b.

          .. versionadded:: 3.6

          Parameters
          ----------
          b: bool

For classes and functions, the directive should be placed before the
*Parameters* section. For parameters, the directive should be placed at the
end of the parameter description. The patch release version is omitted and
the directive should not be added to entire modules.


New modules and files: installation
-----------------------------------

* If you have added new files or directories, or reorganized existing ones, make sure the
  new files are included in the :file:`meson.build` in the corresponding directories.
* New modules *may* be typed inline or using parallel stub file like existing modules.

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

Avoid using pre-computed strings (``f-strings``, ``str.format``,etc.) for logging because
of security and performance issues, and because they interfere with style handlers. For
example, use ``_log.error('hello %s', 'world')``  rather than ``_log.error('hello
{}'.format('world'))`` or ``_log.error(f'hello {s}')``.

Which logging level to use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
.. _gitter: https://gitter.im/matplotlib/matplotlib
.. _community: https://gitter.im/matplotlib/community
