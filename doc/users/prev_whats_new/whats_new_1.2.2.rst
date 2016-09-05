.. _whats-new-1-2-2:

New in matplotlib 1.2.2
=======================

.. contents:: Table of Contents
   :depth: 2



Improved collections
--------------------

The individual items of a collection may now have different alpha
values and be rendered correctly.  This also fixes a bug where
collections were always filled in the PDF backend.

Multiple images on same axes are correctly transparent
------------------------------------------------------

When putting multiple images onto the same axes, the background color
of the axes will now show through correctly.
