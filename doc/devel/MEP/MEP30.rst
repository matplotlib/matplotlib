=============================================
 MEP30: Dummy types for common option selects
=============================================
.. contents::
   :local:


Status
======

- **Discussion**: The MEP is being actively discussed on the mailing
  list and it is being improved by its author.  The mailing list
  discussion of the MEP should include the MEP number (MEPxxx) in the
  subject line so they can be easily related to the MEP.

Branches and Pull requests
==========================

Abstract
========

There is currently a family of matplotlib concepts whose documentation is
contained almost exclusively within the docstrings of functions which take
arguments of that type (and sometimes in tutorials). Common examples are
``linestyle``, ``joinstyle/capstyle``, and ``bounds/extents`` (for a full list,
see below). I will call these not-quite-types "pseudotypes".

As some of these pseudotypes became used by more and more functions, their
documentation has become fractured across the various files that use them. For
example, the ``linestyle`` parameter is accepted in many places, including
the Line2D constructor, Axes methods, various `~.matplotlib.collections`
classes, and of course in rc. While `~.matplotlib.lines.Line2D` fully documents
it only some Axes methods link to this documentation, others simply hint at the
available options.

Input checking for these pseudotypes tends to be repeated across many files and
is too easy to do incorrectly or inconsistently. For example, the ``joinstyle``
and ``capstyle`` parameters have validators in ``rcsetup.py``. However, while
these are used in ``patches.py`` and ``collections.py``, they are not used in
``markers.py``, and ``backend_bases.py`` calls ``cbook._check_in_list`` with its
own list of possible valid joinstyles.

In order to prevent further fragmentation of docs and validation, I propose that
each such concept get a proper new-style class, where we can centralize its
documentation. All functions that accept such an argument will then easily be
able to link to it using the standard numpydoc syntax in their docstrings, and
the description of these parameters can instead be changed to point to relevant
tutorials, instead of an ad-hoc rehashing of already existing documentation.
Error checking would be centralized to that class instead of being scattered
throughout several `cbook._check_in_list` calls that are liable to become stale.

Some benefits of this approach include:

1. Less likely for docs to become stale, due to centralization.
2. Increased discoverability of advanced options. If the simple linestyle option
   ``'-'`` is documented alongside more complex on-off dash specifications,
   users are more likely to scroll down than they are to stumble across an
   unlinked-to tutorial that describes a feature they need.
3. Canonicalization of many of matplotlib's "implicit standards" (like what is a
   "bounds" versus and "extents") that currently have to be learned by reading
   the code.
4. The process would likely highlight issues with API consistency in a way that
   could be more easily tracked via Issues, helping with the process of
   improving our API (see below for discussion).
5. Becoming more compatible with potentially adding typing to the library.
6. Faster doc build times, due to significant decreases in the amount of
   text needing to be parsed.


Detailed description
====================

Historically, matplotlib's API has relied heavily on string-as-enum
"pseudotypes". Besides mimicking matlab's API, these parameter-strings allow the
user to pass semantically-rich values as arguments to matplotlib functions
without having to explicitly import or verbosely prefix an actual enum value
just to pass basic plot options (i.e. ``plt.plot(x, y, linestyle='solid')`` is
easier to type and less redundant than ``plt.plot(x, y,
linestyle=mpl.LineStyle.solid)``).

Many of these string-as-enum pseudotypes have since evolved more sophisticated
features. For example, a ``linestyle`` can now be either a string or a 2-tuple
of sequences, and a MarkerStyle can now be either a string or a path. While this
is true of many pseudotypes, MarkerStyle is the only one (to my knowledge) that
has the status of being a proper Python type.

Because psuedotypes are not classes in their own right, Matplotlib has
historically had to roll its own solutions for centralizing documentation and
validation of these pseudotypes (e.g. the ``docstring.interpd.update`` docstring
interpolation pattern and the ``cbook._check_in_list`` validator pattern,
respectively) instead of using the standard toolchains.

While these solutions have worked well for us, the lack of an explicit location
to document each pseudotype means that the documentation is often difficult to
find, large tables of allowed values are repeated throughout the documentation,
and often an explicit statement of the *scope* of a pseudotype is completely
missing from the docs. Take the ``plt.plot`` docs, for example. In the "Notes",
a description of the matlab-like format-string styling method mentions
``linestyle``, ``color``, and ``markers`` options. There are many more ways to
pass these three values than are hinted at, but, for many users, this is their
only source of understanding about what values are possible for those options
until they stumble on one of the relevant tutorials. In the table of ``Line2D``
attributes, the ``linestyle`` entry does a good job of linking to
``Line2D.set_linestyle`` where those options are described, but the ``color``
and ``markers`` entries do not. ``color`` simply links to ``Line2D.set_color``,
which does nothing in the way of offering intuition on what kinds of inputs are
allowed.

.. It can be argued that ``plt.plot`` is a good candidate to be explicitly
   excempted from any documentation best practices we try to codify, and I've
   chosen it intentionally to elicit the strongest opinions from everyone.

It could be argued that this is something that can be fixed by simply tidying up
the individual docstrings that are causing problems, but the issue is
unfortunately much more systemic than that. Without a centralized place to find
the documentation, this will simply lead to us having more and more copies of
increasingly verbose documentation repeated everywhere each of these pseudotypes
is used. The alternative, of scattering the information throughout the
documentation, will instead lead to the users having to slowly piece together
their mental model of each pseudotype through wiki-diving style traversal
throughout our documentation, or piecemeal from StackOverflow examples.

Ideally, a mention of ``linestyle`` in the ``LineCollection`` docs should
instead link to the same place as it does in the ``plt.plot`` docs. By
organizing these ``linestyle``-specific docs in order from most-common to
most-complex input types, we can maintain a "single-click-to-discover" property
for our advanced plotting options, while also making sure that we don't hurt
usability for users that simply want to know the simplest way to accomplish a
common task.

Practically speaking, the actual information that we want to have in the
``LineCollection`` docs is just:

1. A link to complete docs for allowable inputs (like those found in
   ``Line2D.set_linestyle``).
2. A plain words description of what the parameter is meant to accomplish. To
   matplotlib power users, this is evident from the parameter's name, but for
   new users this need not be the case. (e.g. ``linestyle: a description of
   whether the stroke used to draw each line in the collection is dashed, dotted
   or solid``).
3. A link to any tutorials that visually depict the possible options (currently
   found only after already clicking through to the ``Line2D.set_linestyle``
   docs).

In order to make this information available for all pseudotypes, helping the
continued improval of the consistency and readability of the docs, we propose
the following best-practices for handling pseudotypes:

0. Pseudotype documentation should be centralized at a dedicated class
   definition.
1. Functions that accept pseudotype values should link to the appropriate
   pseudotype class docs.
2. Validation should always happen, but only at the point of usage (i.e.
   immediately before any operation that could raise or produce an error if the
   value is invalid).
3. If a pseudotype is a "string-as-enum", each possible value should have a
   Sphinx-parseable documentation string.
4. If applicable, individual classmethods should be written to construct a
   pseudotype from each of various input possibilities, one per possible input
   type. Obviously, ``__init__`` should delegate to these when possible.

In particular, notice that (1) would replace large copies of
tables of possible linestyles, markerstyles, etc, with links to the complete
documentation for each. Without all the visual noise from these tables of valid
options, the relevant functions would be free to visibly link to tutorials where
these options are visually demonstrated.

This section describes the need for the MEP.  It should describe the
existing problem that it is trying to solve and why this MEP makes the
situation better.  It should include examples of how the new
functionality would be used and perhaps some use cases.

Implementation
==============

This proposal would add one class per pseudotype. For types with complex
construction requirements, we would produce and use classmethods for explicit
construction from a known type, but ``__init__`` would continue to hold the
logic required to deduce how to construct the type from the type of the input.

All functions that accept this pseudotype as a parameter would have their
docstrings changed to simply use the numpydoc "input type" syntax to link to
this new class. All functions which *use* this pseudotype (i.e. would raise on
an invalid input) would construct an explicit object instance using the general
``__init__``, allowing the new class to handle validation.

The pseudotypes that I propose require new style classes are:

1. ``linestyle``
2. ``capstyle``
3. ``joinstyle``
4. ``bounds``
5. ``extents``
6. ``capstyle``

Backward compatibility
======================

This proposal does not break backward compatibility, since the class's
constructor will explicitly be designed to take the same values as were
previously allowed.

Alternatives
============

Instead of making new classes, we can comb through each of the pseudotypes
listed above and choose a single place for the validation to go, documenting
this for discoverability (for example, the only realistic way to discover that
``validate_joinstyle`` exists currently is to ``grep`` for ``joinstyle`` and
find it serendipidously). To fix documentation redundancy, we could use Sphinx's
powerful linking capability to make sure that each pseudotype is only documented
once (by the class that "owns"/validates it), with all other documentation
linking to that location. This pattern would probably require documentation in
the developer docs.
