.. _developers-guide-index:

##########
Contribute
##########

.. ifconfig:: releaselevel != 'dev'

   .. important::

      If you plan to contribute to Matplotlib, please read the
      `development version <https://matplotlib.org/devdocs/devel/index.html>`_
      of this document as it will have the most up to date installation
      instructions, workflow process, and contributing guidelines.

:octicon:`heart;1em;sd-text-info` Thank you for your interest in helping to improve
Matplotlib! :octicon:`heart;1em;sd-text-info`

This project is a community effort, and everyone is welcome to contribute. Everyone
within the community is expected to abide by our :ref:`code of conduct <code_of_conduct>`.

There are various ways to contribute, such as optimizing and refactoring code,
detailing unclear documentation and writing new examples, helping the community,
reporting and fixing bugs, requesting and implementing new features...

.. _submitting-a-bug-report:
.. _request-a-new-feature:

GitHub issue tracker
====================

The `issue tracker <https://github.com/matplotlib/matplotlib/issues>`_ serves as the
centralized location for making feature requests, reporting bugs, identifying major
projects to work on, and discussing priorities.

We have preloaded the issue creation page with markdown forms requesting the information
we need to triage issues and we welcome you to add any additional information or
context that may be necessary for resolving the issue:

.. grid:: 1 1 2 2

   .. grid-item-card::
      :class-header: sd-fs-5

      :octicon:`bug;1em;sd-text-info` **Submit a bug report**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      Thank you for your help in keeping bug reports targeted and descriptive.

      .. button-link:: https://github.com/matplotlib/matplotlib/issues/new/choose
            :expand:
            :color: primary

            Report a bug

   .. grid-item-card::
      :class-header: sd-fs-5

      :octicon:`light-bulb;1em;sd-text-info` **Request a new feature**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      Thank you for your help in keeping feature requests well defined and tightly scoped.

      .. button-link:: https://github.com/matplotlib/matplotlib/issues/new/choose
         :expand:
         :color: primary

         Request a feature

Since Matplotlib is an open source project with limited resources, we encourage users
to also :ref:`participate <contribute_code>` in fixing bugs and implementing new
features.

Contributing guide
==================

We welcome you to get more involved with the Matplotlib project! If you are new
to contributing, we recommend that you first read our
:ref:`contributing guide<contributing>`:

.. toctree::
   :hidden:

   contribute

.. grid:: 1 1 2 2
   :class-row: sd-fs-5 sd-align-minor-center

   .. grid-item::

      .. grid:: 1
         :gutter: 1

         .. grid-item-card::
            :link: contribute_code
            :link-type: ref
            :class-card: sd-shadow-none
            :class-body: sd-text-{primary}

            :octicon:`code;1em;sd-text-info` Contribute code

         .. grid-item-card::
            :link: contribute_documentation
            :link-type: ref
            :class-card: sd-shadow-none
            :class-body: sd-text-{primary}

            :octicon:`note;1em;sd-text-info` Write documentation

         .. grid-item-card::
            :link: contribute_triage
            :link-type: ref
            :class-card: sd-shadow-none
            :class-body: sd-text-{primary}

            :octicon:`issue-opened;1em;sd-text-info` Triage issues

         .. grid-item-card::
            :link: other_ways_to_contribute
            :link-type: ref
            :class-card: sd-shadow-none
            :class-body: sd-text-{primary}

            :octicon:`globe;1em;sd-text-info` Build community

   .. grid-item::

      .. grid:: 1
         :gutter: 1

         .. grid-item::

            :octicon:`info;1em;sd-text-info` :ref:`Is this my first contribution? <new_contributors>`

         .. grid-item::

            :octicon:`question;1em;sd-text-info` :ref:`Where do I ask questions? <get_connected>`

         .. grid-item::

            :octicon:`git-pull-request;1em;sd-text-info` :ref:`How do I choose an issue? <managing_issues_prs>`

         .. grid-item::

            :octicon:`codespaces;1em;sd-text-info` :ref:`How do I start a pull request? <how-to-pull-request>`


.. _development_environment:

Development workflow
====================

If you are contributing code or documentation, please follow our guide for setting up
and managing a development environment and workflow:

.. grid:: 1 1 2 2

   .. grid-item-card::
      :shadow: none

      **Install**
      ^^^

      .. toctree::
         :maxdepth: 2

         development_setup


   .. grid-item-card::
      :shadow: none

      **Workflow**
      ^^^^

      .. toctree::
         :maxdepth: 2

         development_workflow

      .. toctree::
         :maxdepth: 1

         troubleshooting.rst


.. _contribution_guideline:

Policies and guidelines
=======================

These policies and guidelines help us maintain consistency in the various types
of maintenance work. If you are writing code or documentation, following these policies
helps maintainers more easily review your work. If you are helping triage, community
manage, or release manage, these guidelines describe how our current process works.

.. grid:: 1 1 2 2
   :class-row: sf-fs-1
   :gutter: 2

   .. grid-item-card::
      :shadow: none

      **Code**
      ^^^

      .. toctree::
         :maxdepth: 1

         coding_guide
         api_changes
         testing

   .. grid-item-card::
      :shadow: none

      **Documentation**
      ^^^

      .. toctree::
         :maxdepth: 1

         document
         style_guide
         tag_guidelines

   .. grid-item-card::
      :shadow: none

      **Triage And Review**
      ^^^

      .. toctree::
         :maxdepth: 1

         triage
         pr_guide

   .. grid-item-card::
      :shadow: none

      **Maintenance**
      ^^^

      .. toctree::
         :maxdepth: 1

         release_guide
         communication_guide
         min_dep_policy
         MEP/index
