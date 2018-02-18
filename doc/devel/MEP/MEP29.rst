=========================
 MEP29: Text light markup
=========================

.. contents::
   :local:


Status
======

Discussion


Branches and Pull requests
==========================

None at the moment, proof of concept only.

Abstract
========

This MEP proposes to add lightweight markup to the text artist.

Detailed description
====================

Using different size/color/family in a text annotation is difficult because the
`text` method accepts argument for size/color/family/weight/etc. that are used
for the whole text. But, if one wants, for example, to have different colors,
one has to look at the gallery where one such example is provided:
http://matplotlib.org/examples/text_labels_and_annotations/rainbow_text.html

This example takes a list of strings as well as a list of colors which makes it
cumbersome to use. An alternative would be to use a restricted set of pango-like markup (see https://developer.gnome.org/pango/stable/PangoMarkupFormat.html) and to interpret this markup.

Some markup examples::

   Hello <b>world!</b>`
   Hello <span color="blue">world!</span>


Implementation
==============

A proof of concept is provided in `markup_example.py <https://github.com/rougier/matplotlib/blob/markup/examples/text_labels_and_annotations/markup.py>`_ but it currently only handles the horizontal direction.

Improvements
------------

* This proof of concept uses regex to parse the text but it may be better
  to use the html.parser from the standard library.

* Computation of text fragment positions could benefit from the OffsetFrom
  class. See for example item 5 in `Using Complex Coordinates with Annotations <http://matplotlib.org/devdocs/tutorials/text/annotations.html#using-complex-coordinates-with-annotations>`_

Problems
--------

* One serious problem is how to deal with text having both latex and
  html-like tags. For example, consider the following::

     $<b>Bold$</b>

  Recommendation would be to have mutual exclusion.


Backward compatibility
======================

None at the moment since it is only a proof of concept


Alternatives
============

As proposed by @anntzer, this could be also implemented as improvements to
mathtext. For example::

  r"$\text{Hello \textbf{world}}$"
  r"$\text{Hello \textcolor{blue}{world}}$"
  r"$\text{Hello \textsf{\small world}}$"
