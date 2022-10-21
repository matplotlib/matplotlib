=======================================
 MEP24: Negative radius in polar plots
=======================================

.. contents::
   :local:



Status
======
*Discussion*

Branches and Pull requests
==========================

None

Abstract
========

It is clear that polar plots need to be able to gracefully handle
negative r values (not by clipping or reflection).

Detailed description
====================

One obvious application that we should support is bB plots (see
https://github.com/matplotlib/matplotlib/issues/1730#issuecomment-40815837),
but this seems more generally useful (for example growth rate as a
function of angle).  The assumption in the current code (as I
understand it) is that the center of the graph is ``r==0``, however it
would be good to be able to set the center to be at any ``r`` (with any
value less than the offset clipped).

Implementation
==============


Related Issues
==============
#1730, #1603, #2203, #2133



Backward compatibility
======================


Alternatives
============
