Matplotlib documentation
========================

Building the documentation
--------------------------

See :file:`doc/devel/documenting_mpl.rst` for instructions to build the docs.

Organization
------------

This is the top level directory for the Matplotlib
documentation.  All of the documentation is written using Sphinx, a
python documentation system based on reStructuredText.  This directory contains the
following

Files
^^^^^

* index.rst - the top level include document (and landing page) for the Matplotlib docs

* conf.py - the sphinx configuration

* docutils.conf - htmnl output configuration

* Makefile and make.bat - entry points for building the docs

* matplotlibrc - rcParam configuration for docs

* missing-references.json - list of known missing/broken references


Content folders
^^^^^^^^^^^^^^^

* api - templates for generating the api documentation

* devel - documentation for contributing to Matplotlib

* project - about Matplotlib, e.g. mission, code of conduct, licenses, history, etc.

* users - usage documentation, e.g., installation, tutorials, faq, explanations, etc.

* thirdpartypackages - redirect to <https://matplotlib.org/mpl-third-party/>

Build folders
^^^^^^^^^^^^^

* _static - supplementary files; e.g. images, CSS, etc.

* _templates - Sphinx page templates

* sphinxext - Sphinx extensions for the Matplotlib docs

Symlinks
--------

During the documentation build, sphinx-gallery creates symlinks from the source folders
in `/galleries` to target_folders in '/doc'; therefore ensure that you are editing the
real files rather than the symbolic links.

Source files -> symlink:
* galleries/tutorials -> doc/tutorials
* galleries/plot_types -> doc/plot_types
* galleries/examples -> doc/gallery
* galleries/users_explain -> doc/users/explain
