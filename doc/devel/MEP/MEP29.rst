========================
Overhauling the Axes API
========================

.. contents::
   :local:

Abstract
========
The axes API at the moment feels bursting at the seems with so many tweaks and
new features over the years that it has come time for a simplification.  We
have many different types of Axis now, from Polar coordinates, Axes3D, and even
Basemap.

This MEP will simplify the API making our existing classes simpler and pave the
way for more interesting exotic Axes, such as those found in non-Euclidean
geometry, we already implement one such gemoetry, i.e. basemap which implements
the simplest elliptic geometry, and provides various projections for which to
project this geometry onto a 2D surface (the screen/paper).


Detailed description
====================
Before we can look at changing the system, we first need to understand the
concepts of Axes, and this we do in this section.

What do we mean by Axes?
------------------------
Before we can look into the concept of Axes, we need to first look at the
singular, at its simplest level, we think of an Axis as one dimensional,
a simple number line.  It has units and it has a scale.

It contains a coordinate space made of 1 or more `Axis` and we can plot in this
defined space.

Some examples of axes:
+ 2D Cartesian (x, y)
+ 2D Polar (rho, theta)
+ 3D Cartesian (x, y, z)
+ 3D Spherical Polar
+ 3D Cylindrical Polar
+ 2D on a sphere as we have in Basemap

In fact Lagrangian mechanics simplifies the multitude of axes to a generic set
of axis \vec{q}.  As we depend on 2D Cartesian geometry for output to the
screen, and or paper documents etcetera, we thus need to convert from our axes to
2D Cartesian coordinates.

So as not to rewrite all the plot methods for every axes, we need our Base Axes
class to do the conversion for us through an Axes API.

Secondly, we need this API to convert from screen coordinates back to our Axes
coordinates so as to facilitate user interaction.

Finally, we need to stay aware of the fact that coordinate systems do not have
a 1:1 mapping with screen coordinates, that we will specify extra parameters
determined at run time to control this, for example at present the standard 2d
axes uses four parameters to control the extent of the 2D Cartesian screen
domain, set via x_lim, and y_lim these get affected by the pan/zoom controls
of the GUI.  In 3d we have more parameters to control the rotation and zoom.

As well as Axes 3D, Basemap should also welcome this change, with an
anticipated structure of a base mapping class with a coordinate system in
lat/lon coordinates, but with different mapping projections available for the
conversion between the Axes coordinate system and the screen.

Implementation
==============
Axes
----

First we define our coordinate transformation functions:
axes_to_base(self, \*q)
base_to_axes(self, x, y)

The term ``base`` could get replaced with ``screen`` but for now we will keep
it simple to reflect another transformation from base coords to screen coords,
e.g. perhaps to differentiate between window and screen coords.

To provide a common API to interface with the contained ``Axis`` classes, we
will use a dictionary to map the name of the axis to the ``Axis`` class.

We need a view state to obtain the current parameters controlling the
conversion, the main question here lies in whether this should get built into
the axes class directly, or form work as a separate "helper" class, deriving
from a ``ViewStateBase``.  The choice here will determine the class hireachy,
as different 3D axes will have the same view state parameters, but different
transformation methods.  We have three choices here:

1. Direct class hirearchy Base -> 3D -> 3D Specific
2. Class has the parameters class as an attribute
3. Multiple Inheritance, allowing us to mix in these parts, so:
   (Base + View) -> 3D Specific

Axis
----
As mentioned above, an axis basically controls the number side, it has a
scale.  A Cartesian axis can have a linear scale or a log scale, but how about
a polar axis, such an axis due to its periodic nature, we cannot scale.
This suggests an API for the Axis class, we have our base class which deals
with the representation and unit system, and then we have subclasses such as
``CartesianAxis``, and perhaps also ``PolarAxis``.  From here we can define a
generic chain to convert between coordinate systems.  The raw data first gets
scaled by the Axis, before the ``Axes`` class can convert it to base/screen
coordinates.

We should note that a colorbar essentially exists as 1D Cartesian Axis, and
thus it might feel nice for it to also use this class.


Still to think about
====================
+ Twined axes
+ Parasite axes

We need an API to link these more formally together as one.  Imagine having
twin/parasite axes on a 3d plot.  We need a more formal linking both so that
these other axes share the same view state, but also for more general
interaction like reporting the cursor location on all axes that the cursor lies
in.  An AxesContainer would work for the latter but not the former.  We also
need to think about whether we define the colorbar as part of the axes.
Logically speaking it exists as just another axis together with the spatial
ones.

Perhaps we should work this part out later?


Backward Compatibility
======================
So far this new refined API doesn't break the existing API as we can alias the
new API with the old terminology which we can later deprecate if desired.  If
we do deprecate the old API, we would do so over a long period as I imagine
it exists pretty well ingrained in the codebase.
