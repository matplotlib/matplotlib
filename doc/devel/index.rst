.. _developers-guide-index:

##########
Contribute
##########

Thank you for your interest in helping to improve Matplotlib! There are various
ways to contribute: optimizing and refactoring code, detailing unclear
documentation and writing new examples, reporting and fixing bugs and requesting
and implementing new features, helping the community...

New contributors
================

.. grid:: 1 1 2 2
   :class-row: sd-align-minor-center

   .. grid-item::
      :class: sd-fs-5

      :octicon:`info;1em;sd-text-info` :ref:`Where should I start? <start-contributing>`

      :octicon:`question;1em;sd-text-info` :ref:`Where should I ask questions? <get_connected>`

      :octicon:`issue-opened;1em;sd-text-info` :ref:`What are "good-first-issues"? <new_contributors>`

      :octicon:`git-pull-request;1em;sd-text-info` :ref:`How do I claim an issue? <managing_issues_prs>`

      :octicon:`codespaces;1em;sd-text-info` :ref:`How do I start a pull request? <how-to-contribute>`

   .. grid-item::

      .. grid:: 1
         :gutter: 1
         :class-row: sd-fs-5

         .. grid-item-card::
            :link: request-a-new-feature
            :link-type: ref
            :shadow: none

            :octicon:`light-bulb;1em;sd-text-info` Request new feature

         .. grid-item-card::
            :link: submitting-a-bug-report
            :link-type: ref
            :shadow: none

            :octicon:`bug;1em;sd-text-info` Submit bug report

         .. grid-item-card::
            :link: contributing-code
            :link-type: ref
            :shadow: none

            :octicon:`code;1em;sd-text-info` Contribute code

         .. grid-item-card::
            :link: documenting-matplotlib
            :link-type: ref
            :shadow: none

            :octicon:`note;1em;sd-text-info` Write documentation

If you are new to contributing, we recommend that you first read our
:ref:`contributing guide<contributing>`. If you are contributing code or
documentation, please follow our guides for setting up and managing a
:ref:`development environment and workflow<development_environment>`.
For code, documentation, or triage, please follow the corresponding
:ref:`contribution guidelines <contribution_guideline>`.


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

      .. toctree::
         :maxdepth: 1

         dependencies
         ../users/installing/environment_variables_faq.rst


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

      | :ref:`coding_guidelines`

      .. toctree::
         :maxdepth: 1

         coding_guide
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

      **Triage**
      ^^^

      | :ref:`bug_triaging`
      | :ref:`triage_team`
      | :ref:`triage_workflow`

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

.. toctree::
   :hidden:

   contribute
   triage
   license
   color_changes
