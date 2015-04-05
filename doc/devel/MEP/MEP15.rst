==========================================================================
 MEP15 - Fix axis autoscaling when limits are specified for one axis only
==========================================================================

.. contents::
   :local:

Status
======

**Discussion**

Branches and Pull requests
==========================

None so far.

Abstract
========

When one axis of a 2-dimensional plot if overridden via `xlim` or `ylim`,
automatic scaling of the remaining axis should be based on the data that falls
within the specified limits of the first axis.

Detailed description
====================

When axis limits for a 2-D plot are specified for one axis only (via `xlim` or
`ylim`), matplotlib currently does not currently rescale the other axis.  The
result is that the displayed curves or symbols may be compressed into a tiny
portion of the available area, so that the final plot conveys much less
information than it would with appropriate axis scaling.  An example of such a
plot can be found at the following URL:

http://phillipmfeldman.org/Python/MEP15.png

The proposed change of behavior would make matplotlib choose the scale for the
remaining axis using only the data that falls within the limits for the axis
where limits were specified.

Implementation
==============

I don't know enough about the internals of matplotlib to be able to suggest an
implementation.

Backward compatibility
======================

From the standpoint of software interfaces, there would be no break in
backward compatibility.  Some outputs would be different, but if the user
truly desires the previous behavior, he/she can achieve this by overriding
the axis scaling for both axes.

Alternatives
============

The only alternative that I can see is to maintain the status quo.
