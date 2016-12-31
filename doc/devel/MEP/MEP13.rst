=================================
MEP13: Use properties for Artists
=================================

.. contents::
   :local:

Status
======

- **Discussion**

Branches and Pull requests
==========================

None

Abstract
========

Wrap all of the matplotlib getter and setter methods with python
`properties
<https://docs.python.org/3/library/functions.html#property>`_, allowing
them to be read and written like class attributes.

Detailed description
====================

Currently matplotlib uses getter and setter functions (usually
prefixed with get\_ and set\_, respectively) for reading and writing
data related to classes.  However, since 2.6 python supports
properties, which allow such setter and getter functions to be
accessed as though they were attributes.  This proposal would
implement all existing setter and getter methods as properties.

Implementation
==============

1. All existing getter and setter methods will need to have two
   aliases, one with the get\_ or set\_ prefix and one without.
   Getter methods that currently lack prefixes should be recording in
   a text file.
2. Classes should be reorganized so setter and getter methods are
   sequential in the code, with getter methods first.
3. Getter and setter methods the provide additional optional optional
   arguments should have those arguments accessible in another manner,
   either as additional getter or setter methods or attributes of
   other classes.  If those classes are not accessible, getters for
   them should be added.
4. Property decorators will be added to the setter and getter methods
   without the prefix.  Those with the prefix will be marked as
   deprecated.
5. Docstrings will need to be rewritten so the getter with the prefix
   has the current docstring and the getter without the prefix has a
   generic docstring appropriate for an attribute.
6. Automatic alias generation will need to be modified so it will also
   create aliases for the properties.
7. All instances of getter and setter method calls will need to be
   changed to attribute access.
8. All setter and getter aliases with prefixes will be removed

The following steps can be done simultaneously: 1, 2, and 3; 4 and 5;
6 and 7.

Only the following steps must be done in the same release: 4, 5,
and 6.  All other changes can be done in separate releases.  8 should
be done several major releases after everything else.

Backward compatibility
======================

All existing getter methods that do not have a prefix (such as get\_)
will need to be changed from function calls to attribute access.  In
most cases this will only require removing the parenthesis.

setter and getter methods that have additional optional arguments will
need to have those arguments implemented in another way, either as a
separate property in the same class or as attributes or properties of
another class.

Cases where the setter returns a value will need to be changed to
using the setter followed by the getter.

Cases where there are set_ATTR_on() and set_ATTR_off() methods will be
changed to ATTR_on properties.

Examples
========

axes.Axes.set_axis_off/set_axis_on
----------------------------------

Current implementation: ::

   axes.Axes.set_axis_off()
   axes.Axes.set_axis_on()

New implementation: ::

   True = axes.Axes.axis_on
   False = axes.Axes.axis_on
   axes.Axes.axis_on = True
   axes.Axes.axis_on = False

axes.Axes.get_xlim/set_xlim and get_autoscalex_on/set_autoscalex_on
-------------------------------------------------------------------

Current implementation: ::

    [left, right] = axes.Axes.get_xlim()
    auto = axes.Axes.get_autoscalex_on()

    [left, right] = axes.Axes.set_xlim(left=left, right=right, emit=emit, auto=auto)
    [left, right] = axes.Axes.set_xlim(left=left, right=None, emit=emit, auto=auto)
    [left, right] = axes.Axes.set_xlim(left=None, right=right, emit=emit, auto=auto)
    [left, right] = axes.Axes.set_xlim(left=left, emit=emit, auto=auto)
    [left, right] = axes.Axes.set_xlim(right=right, emit=emit, auto=auto)

    axes.Axes.set_autoscalex_on(auto)

New implementation: ::

    [left, right] = axes.Axes.axes_xlim
    auto = axes.Axes.autoscalex_on

    axes.Axes.axes_xlim = [left, right]
    axes.Axes.axes_xlim = [left, None]
    axes.Axes.axes_xlim = [None, right]
    axes.Axes.axes_xlim[0] = left
    axes.Axes.axes_xlim[1] = right

    axes.Axes.autoscalex_on = auto

    axes.Axes.emit_xlim = emit

axes.Axes.get_title/set_title
-----------------------------

Current implementation: ::

    string = axes.Axes.get_title()
    axes.Axes.set_title(string, fontdict=fontdict, **kwargs)

New implementation: ::

    string = axes.Axes.title
    string = axes.Axes.title_text.text

    text.Text = axes.Axes.title_text
    text.Text.<attribute> = attribute
    text.Text.fontdict = fontdict

    axes.Axes.title = string
    axes.Axes.title = text.Text
    axes.Axes.title_text = string
    axes.Axes.title_text = text.Text

axes.Axes.get_xticklabels/set_xticklabels
-----------------------------------------

Current implementation: ::

   [text.Text] = axes.Axes.get_xticklabels()
   [text.Text] = axes.Axes.get_xticklabels(minor=False)
   [text.Text] = axes.Axes.get_xticklabels(minor=True)
   [text.Text] = axes.Axes.([string], fontdict=None, **kwargs)
   [text.Text] = axes.Axes.([string], fontdict=None, minor=False, **kwargs)
   [text.Text] = axes.Axes.([string], fontdict=None, minor=True, **kwargs)

New implementation: ::

   [text.Text] = axes.Axes.xticklabels
   [text.Text] = axes.Axes.xminorticklabels
   axes.Axes.xticklabels = [string]
   axes.Axes.xminorticklabels = [string]
   axes.Axes.xticklabels = [text.Text]
   axes.Axes.xminorticklabels = [text.Text]

Alternatives
============

Instead of using decorators, it is also possible to use the property
function.  This would change the procedure so that all getter methods
that lack a prefix will need to be renamed or removed.  This makes
handling docstrings more difficult and harder to read.

It is not necessary to deprecate the setter and getter methods, but
leaving them in will complicate the code.

This could also serve as an opportunity to rewrite or even remove
automatic alias generation.

Another alternate proposal:

Convert ``set_xlim``, ``set_xlabel``, ``set_title``, etc. to ``xlim``,
``xlabel``, ``title``,... to make the transition from ``plt``
functions to ``axes`` methods significantly simpler. These would still
be methods, not properties, but it's still a great usability
enhancement while retaining the interface.
