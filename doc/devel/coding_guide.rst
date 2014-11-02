.. _coding-guide:

************
Coding guide
************

.. _pull-request-checklist:

Pull request checklist
======================

This checklist should be consulted when creating pull requests to make
sure they are complete before merging.  These are not intended to be
rigidly followed---it's just an attempt to list in one place all of
the items that are necessary for a good pull request.  Of course, some
items will not always apply.

Branch selection
----------------

* In general, simple bugfixes that are unlikely to introduce new bugs
  of their own should be merged onto the maintenance branch.  New
  features, or anything that changes the API, should be made against
  master.  The rules are fuzzy here -- when in doubt, try to get some
  consensus.

  * Once changes are merged into the maintenance branch, they should
    be merged into master.

Style
-----

* Formatting should follow `PEP8
  <http://www.python.org/dev/peps/pep-0008/>`_.  Exceptions to these
  rules are acceptable if it makes the code objectively more readable.

  - You should consider installing/enabling automatic PEP8 checking in your
    editor.  Part of the test suite is checking PEP8 compliance, things
    go smoother if the code is mostly PEP8 compliant to begin with.

* No tabs (only spaces).  No trailing whitespace.

  - Configuring your editor to remove these things upon saving will
    save a lot of trouble.

* Import the following modules using the standard scipy conventions::

    import numpy as np
    import numpy.ma as ma
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.cbook as cbook
    import matplotlib.collections as mcol
    import matplotlib.patches as mpatches

* See below for additional points about
  :ref:`keyword-argument-processing`, if code in your pull request
  does that.

* Adding a new pyplot function involves generating code.  See
  :ref:`new-pyplot-function` for more information.

Documentation
-------------

* Every new feature should be documented.  If it's a new module, don't
  forget to add a new rst file to the API docs.

* Docstrings should be in `numpydoc format
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.
  Don't be thrown off by the fact that many of the existing docstrings
  are not in that format;  we are working to standardize on
  `numpydoc`.

  Docstrings should look like (at a minimum)::

        def foo(bar, baz=None):
            """
            This is a prose description of foo and all the great
            things it does.

            Parameters
            ----------
            bar : (type of bar)
                A description of bar

            baz : (type of baz), optional
                A description of baz

            Returns
            -------
            foobar : (type of foobar)
                A description of foobar
            foobaz : (type of foobaz)
                A description of foobaz
            """
            # some very clever code
            return foobar, foobaz


* Each high-level plotting function should have a simple example in
  the `Example` section of the docstring.  This should be as simple as
  possible to demonstrate the method.  More complex examples should go
  in the `examples` tree.

* Build the docs and make sure all formatting warnings are addressed.

* See :ref:`documenting-matplotlib` for our documentation style guide.

* If your changes are non-trivial, please make an entry in the
  :file:`CHANGELOG`.

* If your change is a major new feature, add an entry to
  :file:`doc/users/whats_new.rst`.

* If you change the API in a backward-incompatible way, please
  document it in :file:`doc/api/api_changes.rst`.

Testing
-------

Using the test framework is discussed in detail in the section
:ref:`testing`.

* If the PR is a bugfix, add a test that fails prior to the change and
  passes with the change.  Include any relevant issue numbers in the
  docstring of the test.

* If this is a new feature, add a test that exercises as much of the
  new feature as possible.  (The `--with-coverage` option may be
  useful here).

* Make sure the Travis tests are passing before merging.

  - The Travis tests automatically test on all of the Python versions
    matplotlib supports whenever a pull request is created or updated.
    The `tox` support in matplotlib may be useful for testing locally.

Installation
------------

* If you have added new files or directories, or reorganized existing
  ones, make sure the new files included in the match patterns in
  :file:`MANIFEST.in`, and/or in `package_data` in `setup.py`.

C/C++ extensions
----------------

* Extensions may be written in C or C++.

* Code style should conform to PEP7 (understanding that PEP7 doesn't
  address C++, but most of its admonitions still apply).

* Interfacing with Python may be done either with the raw Python/C API
  or Cython.  Use of PyCXX is discouraged for new code.

* Python/C interface code should be kept separate from the core C/C++
  code.  The interface code should be named `FOO_wrap.cpp`.

* Header file documentation (aka docstrings) should be in Numpydoc
  format.  We don't plan on using automated tools for these
  docstrings, and the Numpydoc format is well understood in the
  scientific Python community.

Style guide
===========

.. _keyword-argument-processing:

Keyword argument processing
---------------------------

Matplotlib makes extensive use of ``**kwargs`` for pass-through
customizations from one function to another.  A typical example is in
:func:`matplotlib.pylab.text`.  The definition of the pylab text
function is a simple pass-through to
:meth:`matplotlib.axes.Axes.text`::

  # in pylab.py
  def text(*args, **kwargs):
      ret =  gca().text(*args, **kwargs)
      draw_if_interactive()
      return ret

:meth:`~matplotlib.axes.Axes.text` in simplified form looks like this,
i.e., it just passes all ``args`` and ``kwargs`` on to
:meth:`matplotlib.text.Text.__init__`::

  # in axes.py
  def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
      t = Text(x=x, y=y, text=s, **kwargs)

and :meth:`~matplotlib.text.Text.__init__` (again with liberties for
illustration) just passes them on to the
:meth:`matplotlib.artist.Artist.update` method::

  # in text.py
  def __init__(self, x=0, y=0, text='', **kwargs):
      Artist.__init__(self)
      self.update(kwargs)

``update`` does the work looking for methods named like
``set_property`` if ``property`` is a keyword argument.  i.e., no one
looks at the keywords, they just get passed through the API to the
artist constructor which looks for suitably named methods and calls
them with the value.

As a general rule, the use of ``**kwargs`` should be reserved for
pass-through keyword arguments, as in the example above.  If all the
keyword args are to be used in the function, and not passed
on, use the key/value keyword args in the function definition rather
than the ``**kwargs`` idiom.

In some cases, you may want to consume some keys in the local
function, and let others pass through.  You can ``pop`` the ones to be
used locally and pass on the rest.  For example, in
:meth:`~matplotlib.axes.Axes.plot`, ``scalex`` and ``scaley`` are
local arguments and the rest are passed on as
:meth:`~matplotlib.lines.Line2D` keyword arguments::

  # in axes.py
  def plot(self, *args, **kwargs):
      scalex = kwargs.pop('scalex', True)
      scaley = kwargs.pop('scaley', True)
      if not self._hold: self.cla()
      lines = []
      for line in self._get_lines(*args, **kwargs):
          self.add_line(line)
          lines.append(line)

Note: there is a use case when ``kwargs`` are meant to be used locally
in the function (not passed on), but you still need the ``**kwargs``
idiom.  That is when you want to use ``*args`` to allow variable
numbers of non-keyword args.  In this case, python will not allow you
to use named keyword args after the ``*args`` usage, so you will be
forced to use ``**kwargs``.  An example is
:meth:`matplotlib.contour.ContourLabeler.clabel`::

  # in contour.py
  def clabel(self, *args, **kwargs):
      fontsize = kwargs.get('fontsize', None)
      inline = kwargs.get('inline', 1)
      self.fmt = kwargs.get('fmt', '%1.3f')
      colors = kwargs.get('colors', None)
      if len(args) == 0:
          levels = self.levels
          indices = range(len(self.levels))
      elif len(args) == 1:
         ...etc...

Hints
=====

This section describes how to add certain kinds of new features to
matplotlib.

.. _custom_backend:

Developing a new backend
------------------------

If you are working on a custom backend, the *backend* setting in
:file:`matplotlibrc` (:ref:`customizing-matplotlib`) supports an
external backend via the ``module`` directive.  if
:file:`my_backend.py` is a matplotlib backend in your
:envvar:`PYTHONPATH`, you can set use it on one of several ways

* in matplotlibrc::

    backend : module://my_backend

* with the use directive is your script::

    import matplotlib
    matplotlib.use('module://my_backend')

* from the command shell with the -d flag::

    > python simple_plot.py -d module://my_backend


.. _sample-data:

Writing examples
----------------

We have hundreds of examples in subdirectories of
:file:`matplotlib/examples`, and these are automatically generated
when the website is built to show up both in the `examples
<../examples/index.html>`_ and `gallery
<../gallery.html>`_ sections of the website.

Any sample data that the example uses should be kept small and
distributed with matplotlib in the
`lib/matplotlib/mpl-data/sample_data/` directory.  Then in your
example code you can load it into a file handle with::

    import matplotlib.cbook as cbook
    fh = cbook.get_sample_data('mydata.dat')

.. _new-pyplot-function:

Writing a new pyplot function
-----------------------------

A large portion of the pyplot interface is automatically generated by the
`boilerplate.py` script (in the root of the source tree). To add or remove
a plotting method from pyplot, edit the appropriate list in `boilerplate.py`
and then run the script which will update the content in
`lib/matplotlib/pyplot.py`. Both the changes in `boilerplate.py` and
`lib/matplotlib/pyplot.py` should be checked into the repository.

Note: boilerplate.py looks for changes in the installed version of matplotlib
and not the source tree. If you expect the pyplot.py file to show your new
changes, but they are missing, this might be the cause.

Install your new files by running `python setup.py build` and `python setup.py
install` followed by `python boilerplate.py`. The new pyplot.py file should now
have the latest changes.
