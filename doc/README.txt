maptlotlib documentation
========================


Building the documentation
--------------------------

A list of dependencies can be found in ../doc-requirements.txt.

All of these dependencies can be installed through pip::

  pip install -r ../doc-requirements.txt

or conda::

  conda install sphinx numpydoc ipython mock colorspacious pillow

To build the HTML documentation, type ``python make.py html`` in this
directory. The top file of the results will be ./build/html/index.html

**Note that Sphinx uses the installed version of the package to build the
documentation**: matplotlib must be installed *before* the docs can be
generated.

You can build the documentation with several options:

* `--small` saves figures in low resolution.
* `--allowsphinxwarnings`: Don't turn Sphinx warnings into errors.
* `-n N` enables parallel build of the documentation using N process.

Organization
-------------

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

