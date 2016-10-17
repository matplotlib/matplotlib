=========================
 MEPxx: Text light markup
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

One proof of concept is provided in markup_example.py but it only handles
horizontal direction.


Backward compatibility
======================

None at the moment since it is only a proof of concept

Alternatives
============

None (to my knowledge)
