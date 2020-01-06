.. _adding-new-scales:

=========================================================
Developer's guide for creating scales and transformations
=========================================================

.. ::author Michael Droettboom

Matplotlib supports the addition of custom procedures that transform
the data before it is displayed.

There is an important distinction between two kinds of
transformations.  Separable transformations, working on a single
dimension, are called "scales", and non-separable transformations,
that handle data in two or more dimensions at a time, are called
"projections".

From the user's perspective, the scale of a plot can be set with
`.Axes.set_xscale` and `.Axes.set_yscale`.  Projections can be chosen using the
*projection* keyword argument of functions that create Axes, such as
`.pyplot.subplot` or `.pyplot.axes`, e.g. ::

    plt.subplot(projection="custom")

This document is intended for developers and advanced users who need
to create new scales and projections for Matplotlib.  The necessary
code for scales and projections can be included anywhere: directly
within a plot script, in third-party code, or in the Matplotlib source
tree itself.

.. _creating-new-scale:

Creating a new scale
====================

Adding a new scale consists of defining a subclass of
:class:`matplotlib.scale.ScaleBase`, that includes the following
elements:

- A transformation from data coordinates into display coordinates.

- An inverse of that transformation.  This is used, for example, to
  convert mouse positions from screen space back into data space.

- A function to limit the range of the axis to acceptable values
  (``limit_range_for_scale()``).  A log scale, for instance, would
  prevent the range from including values less than or equal to zero.

- Locators (major and minor) that determine where to place ticks in
  the plot, and optionally, how to adjust the limits of the plot to
  some "good" values.  Unlike ``limit_range_for_scale()``, which is
  always enforced, the range setting here is only used when
  automatically setting the range of the plot.

- Formatters (major and minor) that specify how the tick labels
  should be drawn.

Once the class is defined, it must be registered with Matplotlib so
that the user can select it.

A full-fledged and heavily annotated example is in
:doc:`/gallery/scales/custom_scale`.  There are also some classes
in :mod:`matplotlib.scale` that may be used as starting points.


.. _creating-new-projection:

Creating a new projection
=========================

Adding a new projection consists of defining a projection axes which
subclasses :class:`matplotlib.axes.Axes` and includes the following
elements:

- A transformation from data coordinates into display coordinates.

- An inverse of that transformation.  This is used, for example, to
  convert mouse positions from screen space back into data space.

- Transformations for the gridlines, ticks and ticklabels.  Custom
  projections will often need to place these elements in special
  locations, and Matplotlib has a facility to help with doing so.

- Setting up default values (overriding :meth:`~matplotlib.axes.Axes.cla`),
  since the defaults for a rectilinear axes may not be appropriate.

- Defining the shape of the axes, for example, an elliptical axes, that will be
  used to draw the background of the plot and for clipping any data elements.

- Defining custom locators and formatters for the projection.  For
  example, in a geographic projection, it may be more convenient to
  display the grid in degrees, even if the data is in radians.

- Set up interactive panning and zooming.  This is left as an
  "advanced" feature left to the reader, but there is an example of
  this for polar plots in :mod:`matplotlib.projections.polar`.

- Any additional methods for additional convenience or features.

Once the projection axes is defined, it can be used in one of two ways:

- By defining the class attribute ``name``, the projection axes can be
  registered with :func:`matplotlib.projections.register_projection`
  and subsequently simply invoked by name::

      plt.axes(projection='my_proj_name')

- For more complex, parameterisable projections, a generic "projection" object
  may be defined which includes the method ``_as_mpl_axes``. ``_as_mpl_axes``
  should take no arguments and return the projection's axes subclass and a
  dictionary of additional arguments to pass to the subclass' ``__init__``
  method.  Subsequently a parameterised projection can be initialised with::

      plt.axes(projection=MyProjection(param1=param1_value))

  where MyProjection is an object which implements a ``_as_mpl_axes`` method.


A full-fledged and heavily annotated example is in
:doc:`/gallery/misc/custom_projection`.  The polar plot
functionality in :mod:`matplotlib.projections.polar` may also be of
interest.

API documentation
=================

* :mod:`matplotlib.scale`
* :mod:`matplotlib.projections`
* :mod:`matplotlib.projections.polar`
