.. redirect-from:: /devel/document

.. _documenting-matplotlib:
.. _document-build:

*******************
Build Documentation
*******************

All documentation is built from the :file:`doc/`.  The :file:`doc/`
directory contains configuration files for Sphinx and reStructuredText
(ReST_; ``.rst``) files that are rendered to documentation pages.

General file structure
======================
Documentation is created in three ways.  First, API documentation
(:file:`doc/api`) is created by Sphinx_ from
the docstrings of the classes in the Matplotlib library.  Except for
:file:`doc/api/api_changes/`,  ``.rst`` files in :file:`doc/api` are created
when the documentation is built.  See :ref:`writing-docstrings` below.

Second, our example pages, tutorials, and some of the narrative documentation
are created by `Sphinx Gallery`_.  Sphinx Gallery converts example Python files
to ``*.rst`` files with the result of Matplotlib plot calls as embedded images.
See :ref:`writing-examples-and-tutorials` below.

Third, Matplotlib has narrative docs written in ReST_ in subdirectories of
:file:`doc/users/`.  If you would like to add new documentation that is suited
to an ``.rst`` file rather than a gallery or tutorial example, choose an
appropriate subdirectory to put it in, and add the file to the table of
contents of :file:`index.rst` of the subdirectory.  See
:ref:`writing-rest-pages` below.

.. note::

  Don't directly edit the ``.rst`` files in :file:`doc/plot_types`,
  :file:`doc/gallery`,  :file:`doc/tutorials`, and :file:`doc/api`
  (excepting :file:`doc/api/api_changes/`).  Sphinx_ regenerates
  files in these directories when building documentation.

Set up the build
================

The documentation for Matplotlib is generated from reStructuredText (ReST_)
using the Sphinx_ documentation generation tool.

To build the documentation you will need to
:ref:`set up Matplotlib for development <installing_for_devs>`. Note in
particular the :ref:`additional dependencies <doc-dependencies>` required to
build the documentation.

Build the docs
==============

The documentation sources are found in the :file:`doc/` directory.
The configuration file for Sphinx is :file:`doc/conf.py`. It controls which
directories Sphinx parses, how the docs are built, and how the extensions are
used. To build the documentation in html format, cd into :file:`doc/` and run:

.. code-block:: sh

   make html

.. note::

   Since the documentation is very large, the first build may take 10-20 minutes,
   depending on your machine.  Subsequent builds will be faster.

Other useful invocations include

.. code-block:: sh

   # Build the html documentation, but skip generation of the gallery images to
   # save time.
   make html-noplot

   # Build the html documentation, but skip specific subdirectories.  If a gallery
   # directory is skipped, the gallery images are not generated.  The first
   # time this is run, it creates ``.mpl_skip_subdirs.yaml`` which can be edited
   # to add or remove subdirectories
   make html-skip-subdirs

   # Delete built files.  May help if you get errors about missing paths or
   # broken links.
   make clean

   # Build pdf docs.
   make latexpdf

The ``SPHINXOPTS`` variable is set to ``-W --keep-going`` by default to build
the complete docs but exit with exit status 1 if there are warnings.  To unset
it, use

.. code-block:: sh

   make SPHINXOPTS= html

You can use the ``O`` variable to set additional options:

* ``make O=-j4 html`` runs a parallel build with 4 processes.
* ``make O=-Dplot_formats=png:100 html`` saves figures in low resolution.

Multiple options can be combined, e.g. ``make O='-j4 -Dplot_formats=png:100'
html``.

On Windows, set the options as environment variables, e.g.:

.. code-block:: bat

   set SPHINXOPTS= & set O=-j4 -Dplot_formats=png:100 & make html

Show locally built docs
=======================

The built docs are available in the folder :file:`build/html`. A shortcut
for opening them in your default browser is:

.. code-block:: sh

   make show

.. _ReST: https://docutils.sourceforge.io/rst.html
.. _Sphinx: http://www.sphinx-doc.org
.. _`Sphinx Gallery`: https://sphinx-gallery.readthedocs.io/en/latest/
