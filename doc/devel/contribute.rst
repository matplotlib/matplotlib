.. redirect-from:: /devel/contributing

.. _contributing:

**********
Contribute
**********

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

.. dropdown:: Do I really have something to contribute to Matplotlib?
    :open:
    :icon: person-fill

    100% yes! There are so many ways to contribute to our community. Take a look
    at the following sections to learn more.

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

.. _contribute_code:

Code
----
You want to implement a feature or fix a bug or help with maintenance - much
appreciated! Our library source code is found in:

* Python library code: :file:`lib/`
* C-extension code: :file:`src/`
* Tests: :file:`lib/matplotlib/tests/`

Because many people use and work on Matplotlib, we have guidelines for keeping
our code consistent and mitigating the impact of changes.

* :ref:`coding_guidelines`
* :ref:`api_changes`
* :ref:`pr-guidelines`

Code is contributed through pull requests, so we recommend that you start at
:ref:`how-to-pull-request` If you get stuck, please reach out on the
:ref:`contributor_incubator`

.. _contribute_documentation:

Documentation
-------------

You as an end-user of Matplotlib can make a valuable contribution because you
more clearly see the potential for improvement than a core developer. For example,
you can:

- Fix a typo
- Clarify a docstring
- Write or update an :ref:`example plot <gallery>`
- Write or update a comprehensive :ref:`tutorial <tutorials>`

Our code is documented inline in the source code files in :file:`matplotlib/lib`.
Our website structure mirrors our folder structure, meaning that a narrative
document's URL roughly corresponds to its location in our folder structure:

.. grid:: 1 1 2 2

  .. grid-item:: using the library

      * :file:`galleries/plot_types/`
      * :file:`users/getting_started/`
      * :file:`galleries/user_explain/`
      * :file:`galleries/tutorials/`
      * :file:`galleries/examples/`
      * :file:`doc/api/`

  .. grid-item:: information about the library

      * :file:`doc/install/`
      * :file:`doc/project/`
      * :file:`doc/devel/`
      * :file:`doc/users/resources/index.rst`
      * :file:`doc/users/faq.rst`


Other documentation is generated from the following external sources:

* matplotlib.org homepage: https://github.com/matplotlib/mpl-brochure-site
* cheat sheets: https://github.com/matplotlib/cheatsheets
* third party packages: https://github.com/matplotlib/mpl-third-party

Instructions and guidelines for contributing documentation are found in:

* :doc:`document`
* :doc:`style_guide`
* :doc:`tag_guidelines`

Documentation is contributed through pull requests, so we recommend that you start
at :ref:`how-to-pull-request`. If that feels intimidating, we encourage you to
`open an issue`_ describing what improvements you would make. If you get stuck,
please reach out on the :ref:`contributor_incubator`

.. _`open an issue`: https://github.com/matplotlib/matplotlib/issues/new?assignees=&labels=Documentation&projects=&template=documentation.yml&title=%5BDoc%5D%3A+


.. _other_ways_to_contribute:

Community
---------
Matplotlib's community is built by its members, if you would like to help out
see our :ref:`communications-guidelines`.

It helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!

If Matplotlib contributes to a project that leads to a scientific publication,
please cite us following the :doc:`/project/citing` guidelines.

If you have developed an extension to Matplotlib, please consider adding it to our
`third party package <https://github.com/matplotlib/mpl-third-party>`_  list.


.. _get_connected:

Get connected
=============
When in doubt, we recommend going together! Get connected with our community of
active contributors, many of whom felt just like you when they started out and
are happy to welcome you and support you as you get to know how we work, and
where things are.

.. _contributor_incubator:

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

.. _gitter: https://gitter.im/matplotlib/matplotlib
.. _community: https://gitter.im/matplotlib/community


.. _new_contributors:

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

.. _managing_issues_prs:

Work on an issue
================

In general, the Matplotlib project does not assign issues. Issues are
"assigned" or "claimed" by opening a PR; there is no other assignment
mechanism. If you have opened such a PR, please comment on the issue thread to
avoid duplication of work. Please check if there is an existing PR for the
issue you are addressing. If there is, try to work with the author by
submitting reviews of their code or commenting on the PR rather than opening
a new PR; duplicate PRs are subject to being closed.  However, if the existing
PR is an outline, unlikely to work, or stalled, and the original author is
unresponsive, feel free to open a new PR referencing the old one.

.. _good_first_issues:

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


.. _how-to-pull-request:

Start a pull request
====================

The preferred way to contribute to Matplotlib is to fork the `main
repository <https://github.com/matplotlib/matplotlib/>`__ on GitHub,
then submit a "pull request" (PR). You can do this by cloning a copy of the
Maplotlib repository to your own computer, or alternatively using
`GitHub Codespaces <https://docs.github.com/codespaces>`_, a cloud-based
in-browser development environment that comes with the appropriated setup to
contribute to Matplotlib.

Workflow overview
-----------------

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

GitHub Codespaces workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need to open a GUI window with Matplotlib output on Codespaces, our
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

View documentation
""""""""""""""""""

If you also built the documentation pages, you can view them using Codespaces.
Use the "Extensions" icon in the activity bar to install the "Live Server"
extension. Locate the ``doc/build/html`` folder in the Explorer, right click
the file you want to open and select "Open with Live Server."


Open a pull request on Matplotlib
---------------------------------

Finally, go to the web page of *your fork* of the Matplotlib repo, and click
**Compare & pull request** to send your changes to the maintainers for review.
The base repository is ``matplotlib/matplotlib`` and the base branch is
generally ``main``. For more guidance, see GitHub's `pull request tutorial
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_.

For more detailed instructions on how to set up Matplotlib for development and
best practices for contribution, see :ref:`installing_for_devs` and
:ref:`development-workflow`.
