
.. _bug_triaging:

*******************************
Bug triaging and issue curation
*******************************

The `issue tracker <https://github.com/matplotlib/matplotlib/issues>`_
is important to communication in the project because it serves as the
centralized location for making feature requests, reporting bugs,
identifying major projects to work on, and discussing priorities.  For
this reason, it is important to curate the issue list, adding labels
to issues and closing issues that are resolved or unresolvable.

Writing well defined issues increases their chances of being successfully
resolved. Guidelines on writing a good issue can be found in :ref:`here
<submitting-a-bug-report>`. The recommendations in this page are adapted from
the `scikit learn <https://scikit-learn.org/stable/developers/bug_triaging.html>`_
and `Pandas <https://pandas.pydata.org/docs/development/maintaining.html>`_
contributing guides.


Improve issue reports
=====================

Triaging issues does not require any particular expertise in the
internals of Matplotlib, is extremely valuable to the project, and we
welcome anyone to participate in issue triage!  However, people who
are not part of the Matplotlib organization do not have `permissions
to change milestones, add labels, or close issue
<https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-permission-levels-for-an-organization>`_.

If you do not have enough GitHub permissions do something (e.g. add a
label, close an issue), please leave a comment with your
recommendations!

The following actions are typically useful:

- documenting issues that are missing elements to reproduce the problem
  such as code samples

- suggesting better use of code formatting (e.g. triple back ticks in the
  markdown).

- suggesting to reformulate the title and description to make them more
  explicit about the problem to be solved

- linking to related issues or discussions while briefly describing
  how they are related, for instance "See also #xyz for a similar
  attempt at this" or "See also #xyz where the same thing was
  reported" provides context and helps the discussion

- verifying that the issue is reproducible

- classify the issue as a feature request, a long standing bug or a
  regression

.. topic:: Fruitful discussions

   Online discussions may be harder than it seems at first glance, in
   particular given that a person new to open-source may have a very
   different understanding of the process than a seasoned maintainer.

   Overall, it is useful to stay positive and assume good will. `The
   following article
   <http://gael-varoquaux.info/programming/technical-discussions-are-hard-a-few-tips.html>`_
   explores how to lead online discussions in the context of open source.


Maintainers and triage team members
-----------------------------------

In addition to the above, maintainers and the triage team can do the following
important tasks:

- Update labels for issues and PRs: see the list of `available GitHub
  labels <https://github.com/matplotlib/matplotlib/labels>`_.

- Triage issues:

  - **reproduce the issue**, if the posted code is a bug label the issue
    with "status: confirmed bug".

  - **identify regressions**, determine if the reported bug used to
    work as expected in a recent version of Matplotlib and if so
    determine the last working version.  Regressions should be
    milestoned for the next bug-fix release and may be labeled as
    "Release critical".

  - **close usage questions** and politely point the reporter to use
    `discourse <https://discourse.matplotlib.org>`_ or Stack Overflow
    instead and label as "community support".

  - **close duplicate issues**, after checking that they are
    indeed duplicate. Ideally, the original submitter moves the
    discussion to the older, duplicate issue

  - **close issues that cannot be replicated**, after leaving time (at
    least a week) to add extra information


.. topic:: Closing issues: a tough call

    When uncertain on whether an issue should be closed or not, it is
    best to strive for consensus with the original poster, and possibly
    to seek relevant expertise. However, when the issue is a usage
    question or has been considered as unclear for many years, then it
    should be closed.

Preparing PRs for review
========================

Reviewing code is also encouraged. Contributors and users are welcome to
participate to the review process following our :ref:`review guidelines
<pr-guidelines>`.

.. _triage_workflow:

Triage workflow
===============

The following workflow is a good way to approach issue triaging:

#. Thank the reporter for opening an issue

   The issue tracker is many people’s first interaction with the
   Matplotlib project itself, beyond just using the library. As such,
   we want it to be a welcoming, pleasant experience.

#. Is this a usage question? If so close it with a polite message.

#. Is the necessary information provided?

   Check that the poster has filled in the issue template. If crucial
   information (the version of Python, the version of Matplotlib used,
   the OS, and the backend), is missing politely ask the original
   poster to provide the information.

#. Is the issue minimal and reproducible?

   For bug reports, we ask that the reporter provide a minimal
   reproducible example. See `this useful post
   <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_
   by Matthew Rocklin for a good explanation. If the example is not
   reproducible, or if it's clearly not minimal, feel free to ask the reporter
   if they can provide an example or simplify the provided one.
   Do acknowledge that writing minimal reproducible examples is hard work.
   If the reporter is struggling, you can try to write one yourself.

   If a reproducible example is provided, but you see a simplification,
   add your simpler reproducible example.

   If you cannot reproduce the issue, please report that along with your
   OS, Python, and Matplotlib versions.

   If we need more information from either this or the previous step
   please label the issue with "status: needs clarification".

#. Is this a regression?

   While we strive for a bug-free library, regressions are the highest
   priority.  If we have broken user-code that *used to* work, we should
   fix that in the next micro release!

   Try to determine when the regression happened by running the
   reproduction code against older versions of Matplotlib.  This can
   be done by released versions of Matplotlib (to get the version it
   last worked in) or by using `git bisect
   <https://git-scm.com/docs/git-bisect>`_ to find the first commit
   where it was broken.


#. Is this a duplicate issue?

   We have many open issues. If a new issue seems to be a duplicate,
   point to the original issue. If it is a clear duplicate, or consensus
   is that it is redundant, close it. Make sure to still thank the
   reporter, and encourage them to chime in on the original issue, and
   perhaps try to fix it.

   If the new issue provides relevant information, such as a better or
   slightly different example, add it to the original issue as a comment
   or an edit to the original post.

   Label the closed issue with "status: duplicate"

#. Make sure that the title accurately reflects the issue. If you have the
   necessary permissions edit it yourself if it's not clear.

#. Add the relevant labels, such as "Documentation" when the issue is
   about documentation, "Bug" if it is clearly a bug, "New feature" if it
   is a new feature request, ...

   If the issue is clearly defined and the fix seems relatively
   straightforward, label the issue as “Good first issue” (and
   possibly a description of the fix or a hint as to where in the
   code base to look to get started).

   An additional useful step can be to tag the corresponding module e.g.
   the "GUI/Qt" label when relevant.

.. _triage_team:

Triage team
===========


If you would like to join the triage team:

1. Correctly triage 2-3 issues.
2. Ask someone on in the Matplotlib organization (publicly or privately) to
   recommend you to the triage team (look for "Member" on the top-right of
   comments on GitHub).  If you worked with someone on the issues triaged, they
   would be a good person to ask.
3. Responsibly exercise your new power!

Anyone with commit or triage rights may nominate a user to be invited to join
the triage team by emailing matplotlib-steering-council@numfocus.org .
