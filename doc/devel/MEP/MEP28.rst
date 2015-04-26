===================
 MEP28: Text layout
===================

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

Possibly related:

*  https://github.com/matplotlib/matplotlib/issues/1109


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
the graphical description of the layout. It can cover all ``subplot``cases and
should allow to specify moderately complex subplots.

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




Implementation
==============


Backward compatibility
======================



Alternatives
============
