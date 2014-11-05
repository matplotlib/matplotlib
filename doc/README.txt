maptlotlib documentation
========================

This is the top level build directory for the matplotlib
documentation.  All of the documentation is written using sphinx, a
python documentation system built on top of ReST.  This directory contains


* users - the user documentation, e.g., plotting tutorials, configuration
  tips, etc.

* devel - documentation for matplotlib developers

* faq - frequently asked questions

* api - placeholders to automatically generate the api documentation

* mpl_toolkits - documentation of individual toolkits that ship with
  matplotlib

* make.py - the build script to build the html or PDF docs

* index.rst - the top level include document for matplotlib docs

* conf.py - the sphinx configuration

* _static - used by the sphinx build system

* _templates - used by the sphinx build system

* sphinxext - Sphinx extensions for the mpl docs

* mpl_examples - a link to the matplotlib examples in case any
  documentation wants to literal include them

To build the HTML documentation, install sphinx (1.0 or greater
required), then type "python make.py html" in this directory.  Wait
for the initial run (which builds the example gallery) to be done,
then run "python make.py html" again. The top file of the results will
be ./build/html/index.html

Note that Sphinx uses the installed version of the package to build
the documentation, so matplotlib must be installed *before* the docs
can be generated. Even if that is the case, one of the files needed
to do this, '../lib/matplotlib/mpl-data/matplotlibrc', is not version
controlled, but created when matplotlib is built. This means that the
documentation cannot be generated immediately after checking out the
source code, even if matplotlib is installed on your system: you will
have to run ``python setup.py build`` first.

To build a smaller version of the documentation (without
high-resolution PNGs and PDF examples), type "python make.py --small
html".
