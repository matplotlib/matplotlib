.. redirect-from:: /devel/contributing

.. _contributing:

******************
Contributing guide
******************
You've discovered a bug or something else you want to change
in Matplotlib — excellent!

You've worked out a way to fix it — even better!

You want to tell us about it — best of all!

Below, you can find a number of ways to contribute, and how to connect with the
Matplotlib community.

Ways to contribute
==================
.. dropdown:: Do I really have something to contribute to Matplotlib?
    :open:
    :icon: person-fill

    100% yes! There are so many ways to contribute to our community. Take a look
    at the following sections to learn more.

    There are a few typical new contributor profiles:

    * **You are a Matplotlib user, and you see a bug, a potential improvement, or
      something that annoys you, and you can fix it.**

      You can search our `issue tracker <https://github.com/matplotlib/matplotlib/issues>`__
      for an existing issue that describes your problem or
      open a new issue to inform us of the problem you observed and discuss the best approach
      to fix it. If your contributions would not be captured on GitHub (social media,
      communication, educational content), you can also reach out to us on gitter_,
      `Discourse <https://discourse.matplotlib.org/>`__ or attend any of our `community
      meetings <https://scientific-python.org/calendars>`__.

    * **You are not a regular Matplotlib user but a domain expert: you know about
      visualization, 3D plotting, design, technical writing, statistics, or some
      other field where Matplotlib could be improved.**

      Awesome — you have a focus on a specific application and domain and can
      start there. In this case, maintainers can help you figure out the best
      implementation; `open an issue <https://github.com/matplotlib/matplotlib/issues/new/choose>`__
      in our issue tracker, and we'll be happy to discuss technical approaches.

      If you can implement the solution yourself, even better! Consider contributing
      the change as a :ref:`pull request <how-to-pull-request>` right away.

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

You, as an end-user of Matplotlib can make a valuable contribution because you can
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

.. _contribute_triage:

Triage
------
We appreciate your help keeping the `issue tracker <https://github.com/matplotlib/matplotlib/issues>`_
organized because it is our centralized location for feature requests,
bug reports, tracking major projects, and discussing priorities. Some examples of what
we mean by triage are:

* labeling issues and pull requests
* verifying bug reports
* debugging and resolving issues
* linking to related issues, discussion, and external work

Our triage process is discussed in detail in :ref:`bug_triaging`.

If you have any questions about the process, please reach out on the
:ref:`contributor_incubator`

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


.. _generative_ai:


Restrictions on Generative AI Usage
===================================

We expect authentic engagement in our community.

- Do not post output from Large Language Models or similar generative AI as
  comments on GitHub or our discourse server, as such comments tend to be
  formulaic and low content.
- If you use generative AI tools as an aid in developing code or documentation
  changes, ensure that you fully understand the proposed changes and can
  explain why they are the correct approach.

Make sure you have added value based on your personal competency to your
contributions. Just taking some input, feeding it to an AI and posting the
result is not of value to the project. To preserve precious core developer
capacity, we reserve the right to rigorously reject seemingly AI generated
low-value contributions.

In particular, it is also strictly forbidden to post AI generated
content to issues or PRs via automated tooling such as bots or agents. We
may ban such users and/or report them to GitHub.

.. _new_contributors:

New contributors
================

Everyone comes to the project from a different place — in terms of experience
and interest — so there is no one-size-fits-all path to getting involved.  We
recommend looking at existing issue or pull request discussions, and following
the conversations during pull request reviews to get context.  Or you can
deep-dive into a subset of the code-base to understand what is going on.

.. _new_contributors_meeting:

New contributors meeting
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

To join, please go to our public `community gitter`_ channel, and ask to be added to
``#incubator``. One of our core developers will see your message and will add you.

.. _gitter: https://gitter.im/matplotlib/matplotlib
.. _community gitter: https://gitter.im/matplotlib/community

.. _good_first_issues:

Good first issues
-----------------

We have marked some issues as `good first issue
<https://github.com/matplotlib/matplotlib/labels/good%20first%20issue>`_ because we
think they are a good entry point into the process of contributing to Matplotlib. These
issues are well documented, do not require a deep understanding of the internals of
Matplotlib, and do not need urgent resolution. Good first issues are intended to onboard
newcomers with a genuine interest in improving Matplotlib, in the hopes that they will
continue to participate in our development community; therefore, pull requests that are
:ref:`AI generated <generative_ai>` will be closed.

.. _first_contribution:

First contributions
-------------------

If this is your first open source contribution, or your first time contributing to Matplotlib,
and you need help or guidance finding a good first issue, look no further. This section will
guide you through each step:

1. Navigate to the `issues page <https://github.com/matplotlib/matplotlib/issues/>`_.
2. Filter labels with `"Difficulty: Easy" <https://github.com/matplotlib/matplotlib/labels/Difficulty%3A%20Easy>`_
   & `"Good first Issue" <https://github.com/matplotlib/matplotlib/labels/good%20first%20issue>`_ (optional).
3. Click on an issue you would like to work on, and check to see if the issue has a pull request opened to resolve it.

   * A good way to judge if you chose a suitable issue is by asking yourself, "Can I independently submit a PR in 1-2 weeks?"
4. Check existing pull requests (e.g., :ghpull:`28476`) and filter by the issue number to make sure the issue is not in progress:

   * If the issue has a pull request (is in progress), tag the user working on the issue, and ask to collaborate (optional).
   * If there is no pull request, :ref:`create a new pull request <how-to-pull-request>`.
5. Please familiarize yourself with the pull request template (see below),
   and ensure you understand/are able to complete the template when you open your pull request.
   Additional information can be found in the `pull request guidelines <https://matplotlib.org/devdocs/devel/pr_guide.html>`_.

.. dropdown:: `Pull request template <https://github.com/matplotlib/matplotlib/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_
    :open:

    .. literalinclude:: ../../.github/PULL_REQUEST_TEMPLATE.md
       :language: markdown

.. _get_connected:

Get connected
=============

When in doubt, we recommend going together! Get connected with our community of
active contributors, many of whom felt just like you when they started out and
are happy to welcome you and support you as you get to know how we work, and
where things are. You can reach out on any of our :ref:`communication-channels`.
For development questions we recommend reaching out on our development gitter_
chat room and for community questions reach out at `community gitter`_.

.. _managing_issues_prs:

Choose an issue
===============

In general, the Matplotlib project does not assign issues. Issues are
"assigned" or "claimed" by opening a PR; there is no other assignment
mechanism. If you have opened such a PR, please comment on the issue thread to
avoid duplication of work. Please check if there is an existing PR for the
issue you are addressing. If there is, try to work with the author by
submitting reviews of their code or commenting on the PR rather than opening
a new PR; duplicate PRs are subject to being closed.  However, if the existing
PR is an outline, unlikely to work, or stalled, and the original author is
unresponsive, feel free to open a new PR referencing the old one.

Difficulty
----------
Issues may additionally be tagged with a difficulty. ``Difficulty: Easy`` is suitable
for people with beginner scientific Python experience, i.e. fluency with Python syntax
and some experience using libraries like Numpy, Pandas, or Xarray. ``Difficulty: Medium``
and ``Difficulty: Hard`` require more programming experience. This could be for a variety
of reasons, for example:

- requires understanding intermediate to advanced Python features, such as decorators,
  context managers, or meta-programming
- is in areas of the code base which have more interdependencies or is legacy code.
- involves complex or significant changes to algorithms or architecture.

Generally, the difficulty level is correlated with how much conceptual (and contextual)
understanding of Matplotlib is required to resolve it.

.. _how-to-pull-request:

Start a pull request
====================

The preferred way to contribute to Matplotlib is to fork the `main
repository <https://github.com/matplotlib/matplotlib/>`__ on GitHub,
then submit a "pull request" (PR). To work on a pull request:

#. **First** set up a development environment, either by cloning a copy of the
   Matplotlib repository to your own computer or by using Github codespaces, by
   following the instructions in :ref:`installing_for_devs`

#. **Then** start solving the issue, following the guidance in
   :ref:`development workflow <development-workflow>`

#. **As part of verifying your changes** check that your contribution meets
   the :ref:`pull request guidelines <pr-author-guidelines>`
   and then :ref:`open a pull request <open-pull-request>`.

#. **Finally** follow up with maintainers on the PR if waiting more than a few days for
   feedback.  :ref:`Update the pull request <update-pull-request>` as needed.

If you have questions of any sort, reach out on the :ref:`contributor_incubator` and join
the :ref:`new_contributors_meeting`.
