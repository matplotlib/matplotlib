==============================
 MEP10: Docstring consistency
==============================
.. contents::
   :local:

Status
======

**Progress**

Targeted for 1.3

Branches and Pull requests
==========================

#1665
#1757
#1795

Abstract
========

matplotlib has a great deal of inconsistency between docstrings.  This
not only makes the docs harder to read, but it is harder on
contributors, because they don't know which specifications to follow.
There should be a clear docstring convention that is followed
consistently.

The organization of the API documentation is difficult to follow.
Some pages, such as pyplot and axes, are enormous and hard to browse.
There should instead be short summary tables that link to detailed
documentation.  In addition, some of the docstrings themselves are
quite long and contain redundant information.

Building the documentation takes a long time and uses a `make.py`
script rather than a Makefile.

Detailed description
====================

There are number of new tools and conventions available since
matplotlib started using Sphinx that make life easier.  The following
is a list of proposed changes to docstrings, most of which involve
these new features.

Numpy docstring format
----------------------

`Numpy docstring format
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_:
This format divides the docstring into clear sections, each having
different parsing rules that make the docstring easy to read both as
raw text and as HTML.  We could consider alternatives, or invent our
own, but this is a strong choice, as it's well used and understood in
the Numpy/Scipy community.

Cross references
----------------

Most of the docstrings in matplotlib use explicit "roles" when linking
to other items, for example: ``:func:`myfunction```.  As of Sphinx
0.4, there is a "default_role" that can be set to "obj", which will
polymorphically link to a Python object of any type.  This allows one
to write ```myfunction``` instead.  This makes docstrings much easier
to read and edit as raw text.  Additionally, Sphinx allows for setting
a current module, so links like ```~matplotlib.axes.Axes.set_xlim```
could be written as ```~axes.Axes.set_xlim```.

Overriding signatures
---------------------

Many methods in matplotlib use the ``*args`` and ``**kwargs`` syntax
to dynamically handle the keyword arguments that are accepted by the
function, or to delegate on to another function.  This, however, is
often not useful as a signature in the documentation.  For this
reason, many matplotlib methods include something like::

    def annotate(self, *args, **kwargs):
        """
        Create an annotation: a piece of text referring to a data
        point.

        Call signature::

          annotate(s, xy, xytext=None, xycoords='data',
                   textcoords='data', arrowprops=None, **kwargs)
        """

This can't be parsed by Sphinx, and is rather verbose in raw text.  As
of Sphinx 1.1, if the `autodoc_docstring_signature` config value is
set to True, Sphinx will extract a replacement signature from the
first line of the docstring, allowing this::

    def annotate(self, *args, **kwargs):
        """
        annotate(s, xy, xytext=None, xycoords='data',
                   textcoords='data', arrowprops=None, **kwargs)

        Create an annotation: a piece of text referring to a data
        point.
        """

The explicit signature will replace the actual Python one in the
generated documentation.

Linking rather than duplicating
-------------------------------

Many of the docstrings include long lists of accepted keywords by
interpolating things into the docstring at load time.  This makes the
docstrings very long.  Also, since these tables are the same across
many docstrings, it inserts a lot of redundant information in the docs
-- particularly a problem in the printed version.

These tables should be moved to docstrings on functions whose only
purpose is for help.  The docstrings that refer to these tables should
link to them, rather than including them verbatim.

autosummary extension
---------------------

The Sphinx autosummary extension should be used to generate summary
tables, that link to separate pages of documentation.  Some classes
that have many methods (e.g. `Axes.axes`) should be documented with
one method per page, whereas smaller classes should have all of their
methods together.

Examples linking to relevant documentation
------------------------------------------

The examples, while helpful at illustrating how to use a feature, do
not link back to the relevant docstrings.  This could be addressed by
adding module-level docstrings to the examples, and then including
that docstring in the parsed content on the example page.  These
docstrings could easily include references to any other part of the
documentation.

Documentation using help() vs a browser
---------------------------------------

Using Sphinx markup in the source allows for good-looking docs in your
browser, but the markup also makes the raw text returned using help()
look terrible. One of the aims of improving the docstrings should be
to make both methods of accessing the docs look good.

Implementation
==============

1. The numpydoc extensions should be turned on for matplotlib.  There
   is an important question as to whether these should be included in
   the matplotlib source tree, or used as a dependency.  Installing
   Numpy is not sufficient to get the numpydoc extensions -- it's a
   separate install procedure.  In any case, to the extent that they
   require customization for our needs, we should endeavor to submit
   those changes upstream and not fork them.

2. Manually go through all of the docstrings and update them to the
   new format and conventions.  Updating the cross references (from
   ```:func:`myfunc``` to ```func```) may be able to be
   semi-automated.  This is a lot of busy work, and perhaps this labor
   should be divided on a per-module basis so no single developer is
   over-burdened by it.

3. Reorganize the API docs using autosummary and `sphinx-autogen`.
   This should hopefully have minimal impact on the narrative
   documentation.

4. Modify the example page generator (`gen_rst.py`) so that it
   extracts the module docstring from the example and includes it in a
   non-literal part of the example page.

5. Use `sphinx-quickstart` to generate a new-style Sphinx Makefile.
   The following features in the current `make.py` will have to be
   addressed in some other way:

    - Copying of some static content

    - Specifying a "small" build (only low-resolution PNG files for examples)

Steps 1, 2, and 3 are interdependent.  4 and 5 may be done
independently, though 5 has some dependency on 3.

Backward compatibility
======================

As this mainly involves docstrings, there should be minimal impact on
backward compatibility.

Alternatives
============

None yet discussed.
