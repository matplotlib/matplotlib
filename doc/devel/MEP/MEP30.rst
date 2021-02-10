============================================
 MEP30: Axes for downsampled line2d plotting 
============================================

.. contents::
   :local:


Status
======

- **Discussion**: This MEP has not commenced yet, but here are some
  ongoing ideas which may become a part of this MEP:


Branches and Pull requests
==========================


Abstract
========

Provide axes which plot line2d objects in a downsampled fashion, 
in order to speed up plotting of lines with many data points.

Detailed description
====================

Probably the most often used plot objects are line2d objects for xy-plotting. 
If the underlying data has many points and several lines have to be plotted together 
in a single figure or spread over several figures, the plot performance can degrade 
drastically, e.g. when zooming into such axes. 

Downsampled line plotting plots only a subset of all points, but should try to keep the main 
features of the plotted data (maxima/minima should not be lost by downsamplng, some intermediate 
data points should also be plotted, eg. if we plot noisy data, etc.)


Implementation
==============

See this [first attempt](https://github.com/EBenkler/matplotlib-dsaxes), which is far from complete (especially regarding documentation and 
probably conformity with Matplotlib rules as those concerning tests). 
The dsaxes module defines several classes, among them one which can be used to 
dynamically substitute Axes.axes with a dsaxes class that allows downsampled plotting. 
For further details, see readme.md in the above pull request.


Backward compatibility
======================

Not known, exploratory-tested with Matplotlib 3.3.4

Alternatives
============

None known.
