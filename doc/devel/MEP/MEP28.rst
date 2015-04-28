=======================================================
 MEP28: Semi-graphical specification of subplots layout
=======================================================

.. contents::
   :local:


Status
======

**Discussion**

.. - **Discussion**: The MEP is being actively discussed on the mailing
..   list and it is being improved by its author.  The mailing list
..   discussion of the MEP should include the MEP number (MEPxxx) in the
..   subject line so they can be easily related to the MEP.

.. - **Progress**: Consensus was reached on the mailing list and
..   implementation work has begun.

.. - **Completed**: The implementation has been merged into master.

.. - **Superseded**: This MEP has been abandoned in favor of another
     approach.

Branches and Pull requests
==========================

* `Issue 1109 <https://github.com/matplotlib/matplotlib/issues/1109>`_
* `Initial thread (mailing list) <https://www.mail-archive.com/matplotlib-devel%40lists.sourceforge.net/msg11325.html>`_
* `MEP28 PR <https://github.com/matplotlib/matplotlib/pull/4384>`_


Abstract
========

This MEP proposes a new graphical way of specifying simple to moderately
complex subplot layouts using arrays of strings.


Detailed description
====================

The most popular way of specifying subplots layout is to use the ``subplot``
function. This ``subplot`` function is rather unintuitive and makes it
difficult to specify moderately complex subplots. There are alternatives, some
of them very powerful, but they generally require a lot of code to specifiy
precisely the layout. There might be room for an in-between method that rely on
the graphical description of the layout. It can cover all ``subplot`` cases and
should allow to specify moderately complex subplots.

Examples
--------

::

  "AB"
  ┌────────┐┌────────┐
  │ A      ││ B      │
  │        ││        │
  │        ││        │
  └────────┘└────────┘

  "ABB"
  ┌──────┐┌──────────┐
  │ A    ││ B        │
  │      ││          │
  │      ││          │
  └──────┘└──────────┘

  "ABD"
  "CCD"
  ┌───────┐┌───────┐┌───────┐
  │ A     ││ B     ││ D     │
  │       ││       ││       │
  │       ││       ││       │
  └───────┘└───────┘│       │
  ┌────────────────┐│       │
  │ C              ││       │
  │                ││       │
  │                ││       │
  └────────────────┘└───────┘

  "AaBb"
  ┌───────┐┌─┐┌───────┐┌─┐
  │ A     ││ ││ B     ││ │
  │       ││ ││       ││ │
  │       ││ ││       ││ │
  └───────┘└─┘└───────┘└─┘

  "  b  "
  "aABCc"
              ┌───────┐
              └───────┘
  ┌─┐┌───────┐┌───────┐┌───────┐┌─┐
  │ ││ A     ││ B     ││ C     ││ │
  │ ││       ││       ││       ││ │
  │ ││       ││       ││       ││ │
  └─┘└───────┘└───────┘└───────┘└─┘


Syntax
------

A subplot is described using a capital letter, from ``A`` to ``Z`` such that
only 26 subplots can be specified at once. Colorbars are specified using
non-capital letters, from ``a`` to ``z``. The main command is the ``layout``
command that returns a list of axes (subplots and colorbars) following the
lexicographic order, mixing capital and small letters. For example::

  A,B,C = layout("ABC")
  A,B,b = layout("ABb")

A layout can span several lines. In such a case, it is necessary the specify the
layout using an array of strings::

  A,B = layout(["A",
                "B"])

The size of a subplot is relative to the size of its string representation, a
single letter representing the logical unit. For example::


  A,B = layout("AAB")

means ``A`` is two logical block wide while ``B`` is one logical block
wide. The final result should be ``A`` width to be 2/3 while ``B`` width should
be 1/3.

Colorbars can be specified using a small letter and has either a fixed width
(vertical colorbar) or a fixed height (horizontal colorbar). For example::

  A,a = layout("Aa")

The exact width or height of colorbars must be specified throught matplotlib
properties. The difficulty being the attachment of the colorbar to the right
figure. In the example above, the size of ``A`` should be logical block and the
colorbar ``a`` should be attached to the right side of ``A``.

  A,a,B = layout("AaB")

In such a case, ``A`` and ``B`` size is one logical block and ``a`` should be
attached to the right side of ``A``.




Subplot aspects
---------------

It is not yet clear how subplot aspects can be specified and how this
would constrain the overall layout. We need more examples.


Implementation
==============



**Note**: Implementation probably requires a geometry manager able to enforce
constraints but it is not yet clear if a full geometry manager is required
(constraints might be relatively easy to solve).


Backward compatibility
======================

No backward compatibility since this MEP proposes a new method.


Alternatives
============

* `Axes <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axes>`_
* `Subplot <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot>`_
* `GridSpec <http://matplotlib.org/users/gridspec.html>`_
