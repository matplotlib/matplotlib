Overview of the matplotlib src tree
===================================

This is the source directory for matplotlib, which contains the
following files and directories.

* doc - the matplotlib documentation.  See doc/users for the user's
  documentation and doc/devel for the developers documentation

* examples - a bunch of examples using matplotib.  See
  examples/README.txt for information

* setup.cfg.template - used to configure the matplotlib build process.
  Copy this file to setup.cfg if you want to override the default
  build behavior

* matplotlibrc.template - a template file used to generate the
  matplotlibrc config file at build time.  The matplotlibrc file will
  be installed in matplotlib/mpl-data/matplotlibrc

* lib - the python src code.  matplotlib ships several third party
  packages here.  The subdirectory lib/matplotlib contains the python
  src code for matplotlib

* src - the matplotlib extension code, mostly C++

* ttconv - some truetype font utilities

* license - all the licenses for code included with matplotlib.
  matplotlib uses only BSD compatible code

* unit - some unit tests

* CHANGELOG - all the significant changes to matplotlib, organized by
  release.  The top of this file will show you the most recent changes

* API_CHANGES - any change that alters the API is listed here.  The
  entries are organized by release, with most recent entries first

* MIGRATION.txt - instructions on moving from the 0.91 code to the
  0.98 trunk.

* SEGFAULTS - some tips for how to diagnose and debug segfaults

* setup.py - the matplotlib build script

* setupext.py - some helper code for setup.py to build the matplotlib
  extensions

* boilerplate.py - some code to automatically generate the pyplot
  wrappers

* DEVNOTES - deprecated developer notes.  TODO: update and move to the
  doc/devel framework

* FILETYPES - This is a table of the output formats supported by each
  backend.  TODO: move to doc/users

* INTERACTIVE - instructions on using matplotlib interactively, eg
  from the python shell.  TODO: update and move to doc/users.

