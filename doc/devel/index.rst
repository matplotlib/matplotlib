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

There are various ways to contribute: optimizing and refactoring code, detailing
unclear documentation and writing new examples, helping the community, reporting
and fixing bugs and requesting and implementing new features...

.. _submitting-a-bug-report:
.. _request-a-new-feature:

.. grid:: 1 1 2 2

   .. grid-item-card::
      :class-header: sd-fs-5

      :octicon:`bug;1em;sd-text-info` **Submit a bug report**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      We have preloaded the issue creation page with a Markdown form that you can
      use to provide relevant context. Thank you for your help in keeping bug reports
      complete, targeted and descriptive.

      .. button-link:: https://github.com/matplotlib/matplotlib/issues/new/choose
            :expand:
            :color: primary

            Report a bug

   .. grid-item-card::
      :class-header: sd-fs-5

      :octicon:`light-bulb;1em;sd-text-info` **Request a new feature**
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      We will give feedback on the feature proposal. Since
      Matplotlib is an open source project with limited resources, we encourage
      users to then also :ref:`participate in the implementation <contribute_code>`.

      .. button-link:: https://github.com/matplotlib/matplotlib/issues/new/choose
         :expand:
         :color: primary

         Request a feature


We welcome you to get more involved with the Matplotlib project! If you are new
to contributing, we recommend that you first read our
:ref:`contributing guide<contributing>`. If you are contributing code or
documentation, please follow our guides for setting up and managing a
:ref:`development environment and workflow<development_environment>`.
For code, documentation, or triage, please follow the corresponding
:ref:`contribution guidelines <contribution_guideline>`.

New contributors
================

.. toctree::
   :hidden:

   contribute

.. grid:: 1 1 2 2
   :class-row: sd-align-minor-center

   .. grid-item::
      :class: sd-fs-5

      :octicon:`info;1em;sd-text-info` :ref:`Where should I start? <start-contributing>`

      :octicon:`question;1em;sd-text-info` :ref:`Where should I ask questions? <get_connected>`

      :octicon:`git-pull-request;1em;sd-text-info` :ref:`How do I work on an issue? <managing_issues_prs>`

      :octicon:`codespaces;1em;sd-text-info` :ref:`How do I start a pull request? <how-to-pull-request>`

   .. grid-item::

      .. grid:: 1
         :gutter: 1
         :class-row: sd-fs-5

         .. grid-item-card::
            :link: contribute_code
            :link-type: ref
            :shadow: none

            :octicon:`code;1em;sd-text-info` Contribute code

         .. grid-item-card::
            :link: contribute_documentation
            :link-type: ref
            :shadow: none

            :octicon:`note;1em;sd-text-info` Write documentation

         .. grid-item-card::
            :link: other_ways_to_contribute
            :link-type: ref
            :shadow: none

            :octicon:`globe;1em;sd-text-info` Build community




.. _development_environment:

Development environment
=======================

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
