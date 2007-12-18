===============================================
Adding new scales and projections to matplotlib
===============================================

.. ::author Michael Droettboom

Matplotlib supports the addition of new transformations that transform
the data before it is displayed.  Separable transformations, that work
on a single dimension are called "scales", and non-separable
transformations, that take data in two or more dimensions as input are
called "projections".

This document is intended for developers and advanced users who need
to add more scales and projections to matplotlib.

From the user's perspective, the scale of a plot can be set with
``set_xscale`` and ``set_yscale``.  Choosing the projection
currently has no *standardized* method. [MGDTODO]

Creating a new scale
====================

Adding a new scale consists of defining a subclass of ``ScaleBase``,
that brings together the following elements:

  - A transformation from data space into plot space.

  - An inverse of that transformation.  For example, this is used to
    convert mouse positions back into data space.

  - A function to limit the range of the axis to acceptable values.  A
    log scale, for instance, would prevent the range from including
    values less than or equal to zero.

  - Locators (major and minor) that determine where to place ticks in
    the plot, and optionally, how to adjust the limits of the plot to
    some "good" values.

  - Formatters (major and minor) that specify how the tick labels
    should be drawn.

There are a number of ``Scale`` classes in ``scale.py`` that may be
used as starting points for new scales.  As an example, this document
presents adding a new scale ``MercatorLatitudeScale`` which can be
used to plot latitudes in a Mercator_ projection.  For simplicity,
this scale assumes that it has a fixed center at the equator.  The
code presented here is a simplification of actual code in
``matplotlib``, with complications added only for the sake of
optimization removed.

First define a new subclass of ``ScaleBase``::

    class MercatorLatitudeScale(ScaleBase):
        """
        Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
        the system used to scale latitudes in a Mercator projection.

        The scale function:
          ln(tan(y) + sec(y))

        The inverse scale function:
          atan(sinh(y))

        Since the Mercator scale tends to infinity at +/- 90 degrees,
        there is user-defined threshold, above and below which nothing
        will be plotted.  This defaults to +/- 85 degrees.

        source:
        http://en.wikipedia.org/wiki/Mercator_projection
        """
        name = 'mercator_latitude'

This class must have a member ``name`` that defines the string used to
select the scale.  For example,
``gca().set_yscale("mercator_latitude")`` would be used to select the
Mercator latitude scale.

Next define two nested classes: one for the data transformation and
one for its inverse.  Both of these classes must be subclasses of
``Transform`` (defined in ``transforms.py``).::

        class MercatorLatitudeTransform(Transform):
            input_dims = 1
            output_dims = 1

There are two class-members that must be defined.  ``input_dims`` and
``output_dims`` specify number of input dimensions and output
dimensions to the transformation.  These are used by the
transformation framework to do some error checking and prevent
incompatible transformations from being connected together.  When
defining transforms for a scale, which are by definition separable and
only have one dimension, these members should always be 1.

``MercatorLatitudeTransform`` has a simple constructor that takes and
stores the *threshold* for the Mercator projection (to limit its range
to prevent plotting to infinity).::

            def __init__(self, thresh):
                Transform.__init__(self)
                self.thresh = thresh

The ``transform`` method is where the real work happens: It takes an N
x 1 ``numpy`` array and returns a transformed copy.  Since the range
of the Mercator scale is limited by the user-specified threshold, the
input array must be masked to contain only valid values.
``matplotlib`` will handle masked arrays and remove the out-of-range
data from the plot.  Importantly, the transformation should return an
array that is the same shape as the input array, since these values
need to remain synchronized with values in the other dimension.::

            def transform(self, a):
                masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)
                return ma.log(ma.abs(ma.tan(masked) + 1.0 / ma.cos(masked)))

Lastly for the transformation class, define a method to get the
inverse transformation::

            def inverted(self):
                return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(self.thresh)

The inverse transformation class follows the same pattern, but
obviously the mathematical operation performed is different::

        class InvertedMercatorLatitudeTransform(Transform):
            input_dims = 1
            output_dims = 1

            def __init__(self, thresh):
                Transform.__init__(self)
                self.thresh = thresh

            def transform(self, a):
                return npy.arctan(npy.sinh(a))

            def inverted(self):
                return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)

Now we're back to methods for the ``MercatorLatitudeScale`` class.
Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
be passed along to the scale's constructor.  In the case of
``MercatorLatitudeScale``, the ``thresh`` keyword argument specifies
the degree at which to crop the plot data.  The constructor also
creates a local instance of the ``Transform`` class defined above,
which is made available through its ``get_transform`` method::

        def __init__(self, axis, **kwargs):
            thresh = kwargs.pop("thresh", (85 / 180.0) * npy.pi)
            if thresh >= npy.pi / 2.0:
                raise ValueError("thresh must be less than pi/2")
            self.thresh = thresh
            self._transform = self.MercatorLatitudeTransform(thresh)

        def get_transform(self):
            return self._transform

The ``limit_range_for_scale`` method must be provided to limit the
bounds of the axis to the domain of the function.  In the case of
Mercator, the bounds should be limited to the threshold that was
passed in.  Unlike the autoscaling provided by the tick locators, this
range limiting will always be adhered to, whether the axis range is set
manually, determined automatically or changed through panning and
zooming::

        def limit_range_for_scale(self, vmin, vmax, minpos):
            return max(vmin, -self.thresh), min(vmax, self.thresh)

Lastly, the ``set_default_locators_and_formatters`` method sets up the
locators and formatters to use with the scale.  It may be that the new
scale requires new locators and formatters.  Doing so is outside the
scope of this document, but there are many examples in ``ticker.py``.
The Mercator example uses a fixed locator from -90 to 90 degrees and a
custom formatter class to put convert the radians to degrees and put a
degree symbol after the value::

        def set_default_locators_and_formatters(self, axis):
            class DegreeFormatter(Formatter):
                def __call__(self, x, pos=None):
                    # \u00b0 : degree symbol
                    return u"%d\u00b0" % ((x / npy.pi) * 180.0)

            deg2rad = npy.pi / 180.0
            axis.set_major_locator(FixedLocator(
                    npy.arange(-90, 90, 10) * deg2rad))
            axis.set_major_formatter(DegreeFormatter())
            axis.set_minor_formatter(DegreeFormatter())

Now that the Scale class has been defined, it must be registered so
that ``matplotlib`` can find it::

       register_scale(MercatorLatitudeScale)

.. _Mercator: http://en.wikipedia.org/wiki/Mercator_projection