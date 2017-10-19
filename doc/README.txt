Matplotlib documentation
========================


Building the documentation
--------------------------

To build the documentation, you will need additional dependencies:

* Sphinx-1.3 or later (version 1.5.0 is not supported)
* numpydoc 0.4 or later
* IPython
* mock
* colorspacious
* pillow
* graphviz

All of these dependencies *except graphviz* can be installed through pip::

  pip install -r ../doc-requirements.txt

or all of them via conda and pip::

  conda install sphinx numpydoc ipython mock graphviz pillow \
    sphinx-gallery
  pip install colorspacious

To build the HTML documentation, type ``make html`` in this
directory. The top file of the results will be ./build/html/index.html

**Note that Sphinx uses the installed version of the package to build the
documentation**: Matplotlib must be installed *before* the docs can be
generated.

You can build the documentation with several options:

* `make html-noplot` doesn't save the gallery's images. Allows for fast build.
* `make html-allow-warnings`: Don't turn Sphinx warnings into errors.

Organization
-------------

This is the top level build directory for the Matplotlib
documentation.  All of the documentation is written using sphinx, a
python documentation system built on top of ReST.  This directory contains

* users - the user documentation, e.g., plotting tutorials, configuration
  tips, etc.

* devel - documentation for Matplotlib developers

* faq - frequently asked questions

* api - placeholders to automatically generate the api documentation

* mpl_toolkits - documentation of individual toolkits that ship with
  Matplotlib

* Makefile - the build script to build the html or PDF docs

* index.rst - the top level include document for Matplotlib docs

* conf.py - the sphinx configuration

* _static - used by the sphinx build system

* _templates - used by the sphinx build system

* sphinxext - Sphinx extensions for the mpl docs

* mpl_examples - a link to the Matplotlib examples in case any
  documentation wants to literal include them
