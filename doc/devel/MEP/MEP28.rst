=======================================================
 MEP28: Semi-graphical specification of subplots layout
=======================================================

.. contents::
   :local:


Status
======

**Discussion**

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
command that returns an ordered dictionary of axes (subplots and colorbars)
following the lexicographic order, mixing capital and small letters. For
example::

  A,B,C = layout("ABC").values()
  A,B,b = layout("ABb").values()

A layout can span several lines. In such a case, it is necessary the specify the
layout using an array of strings::

  A,B = layout("A", "B").values

The size of a subplot is relative to the size of its string representation, a
single letter representing the logical unit. For example::


  A,B = layout("AAB").values()

means ``A`` is two logical block wide while ``B`` is one logical block
wide. The final result should be ``A`` width to be 2/3 while ``B`` width should
be 1/3.

Colorbars can be specified using a small letter and has either a fixed width
(vertical colorbar) or a fixed height (horizontal colorbar). For example::

  A,a = layout("Aa").values()

The exact width or height of colorbars must be specified throught matplotlib
properties. The difficulty being the attachment of the colorbar to the right
figure. In the example above, the size of ``A`` should be logical block and the
colorbar ``a`` should be attached to the right side of ``A``::

  A,a,B = layout("AaB").values()

In such a case, ``A`` and ``B`` size is one logical block and ``a`` should be
attached to the right side of ``A``.


UTF8 handling
-------------

Characters from any language can actually be used but this might prevent the
distinction between capital and small letters for some language.


Subplot aspects
---------------

It is not yet clear how a subplot aspect can be specified and how this would
constrain the overall layout. We need more user-cases.


Implementation
==============

The simplest cases (without colorbars) can be implemented using `GridSpec
<http://matplotlib.org/users/gridspec.html>`_ using the code proposed by
Till Stensitzki::

  import matplotlib.pyplot as plt
  def parase_layout(s):
      s = [list(i) for i in s]
      s_T = transpose_list(s)
      n_cols = len(s[0])
      n_rows = len(s)
      added_axes = {}
      for i, row in enumerate(s):
          for j, ch in enumerate(row):
              if not ch in added_axes:
                  width = n_rows-row.index(ch)-row[::-1].index(ch)
                  height = n_cols-s_T[j].index(ch)-s_T[j][::-1].index(ch)
                  added_axes[ch] = ((i,j), width, height)
      return added_axes, (n_rows, n_cols)

  def transpose_list(l):
      return map(list, zip(*l))

  def layout(s):
      d, (nr, nc) = parase_layout(s)
      gs = plt.GridSpec(nr, nc)
      for pos, width, height in d.values():
          t = gs[pos[0]:pos[0]+height, pos[1]:pos[1]+width]
          plt.subplot(t)

  s = ['AAB', 'AAB', 'CCC']
  layout(s)

Taking colorbars into account may require a more elaborated approach (geometry
manager) in order to enforce the different constraints.


Backward compatibility
======================

No backward compatibility since this MEP proposes a new method.


Alternatives
============

* `Axes <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axes>`_
* `Subplot <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot>`_
* `GridSpec <http://matplotlib.org/users/gridspec.html>`_
