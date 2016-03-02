Writing code for Python 2 and 3
-------------------------------

As of matplotlib 1.4, the `six <http://pythonhosted.org/six/>`_
library is used to support Python 2 and 3 from a single code base.
The `2to3` tool is no longer used.

This document describes some of the issues with that approach and some
recommended solutions.  It is not a complete guide to Python 2 and 3
compatibility.

Welcome to the ``__future__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The top of every `.py` file should include the following::

    from __future__ import (absolute_import, division,
                            print_function, unicode_literals)
    import six

This will make the Python 2 interpreter behave as close to Python 3 as
possible.

All matplotlib files should also import `six`, whether they are using
it or not, just to make moving code between modules easier, as `six`
gets used *a lot*.


Finding places to use six
^^^^^^^^^^^^^^^^^^^^^^^^^

The only way to make sure code works on both Python 2 and 3 is to make sure it
is covered by unit tests.

However, the `2to3` commandline tool can also be used to locate places
that require special handling with `six`.

(The `modernize <https://pypi.python.org/pypi/modernize>`_ tool may
also be handy, though I've never used it personally).

The `six <http://pythonhosted.org/six/>`_ documentation serves as a
good reference for the sorts of things that need to be updated.

The dreaded ``\u`` escapes
^^^^^^^^^^^^^^^^^^^^^^^^^^

When `from __future__ import unicode_literals` is used, all string
literals (not preceded with a `b`) will become unicode literals.

Normally, one would use "raw" string literals to encode strings that
contain a lot of slashes that we don't want Python to interpret as
special characters.  A common example in matplotlib is when it deals
with TeX and has to represent things like ``r"\usepackage{foo}"``.
Unfortunately, on Python 2there is no way to represent `\u` in a raw
unicode string literal, since it will always be interpreted as the
start of a unicode character escape, such as `\u20af`.  The only
solution is to use a regular (non-raw) string literal and repeat all
slashes, e.g. ``"\\usepackage{foo}"``.

The following shows the problem on Python 2::

    >>> ur'\u'
      File "<stdin>", line 1
    SyntaxError: (unicode error) 'rawunicodeescape' codec can't decode bytes in
    position 0-1: truncated \uXXXX
    >>> ur'\\u'
    u'\\\\u'
    >>> u'\u'
      File "<stdin>", line 1
    SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in
    position 0-1: truncated \uXXXX escape
    >>> u'\\u'
    u'\\u'

This bug has been fixed in Python 3, however, we can't take advantage
of that and still support Python 2::

    >>> r'\u'
    '\\u'
    >>> r'\\u'
    '\\\\u'
    >>> '\u'
      File "<stdin>", line 1
    SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in
    position 0-1: truncated \uXXXX escape
    >>> '\\u'
    '\\u'

Iteration
^^^^^^^^^

The behavior of the methods for iterating over the items, values and
keys of a dictionary has changed in Python 3.  Additionally, other
built-in functions such as `zip`, `range` and `map` have changed to
return iterators rather than temporary lists.

In many cases, the performance implications of iterating vs. creating
a temporary list won't matter, so it's tempting to use the form that
is simplest to read.  However, that results in code that behaves
differently on Python 2 and 3, leading to subtle bugs that may not be
detected by the regression tests.  Therefore, unless the loop in
question is provably simple and doesn't call into other code, the
`six` versions that ensure the same behavior on both Python 2 and 3
should be used.  The following table shows the mapping of equivalent
semantics between Python 2, 3 and six for `dict.items()`:

============================== ============================== ==============================
Python 2                       Python 3                       six
============================== ============================== ==============================
``d.items()``                  ``list(d.items())``            ``list(six.iteritems(d))``
``d.iteritems()``              ``d.items()``                  ``six.iteritems(d)``
============================== ============================== ==============================

Numpy-specific things
^^^^^^^^^^^^^^^^^^^^^

When specifying dtypes, all strings must be byte strings on Python 2
and unicode strings on Python 3.  The best way to handle this is to
force cast them using `str()`.  The same is true of structure
specifiers in the `struct` built-in module.
