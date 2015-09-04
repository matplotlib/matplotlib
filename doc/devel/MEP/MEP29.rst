Overhauling Axes

What do we mean by Axes?
------------------------

It contains a coordinate space made of 1 or more `Axis` and we can plot in this
defined space.

Some examples of axes:
2D Cartesian (x, y)
2D Polar (rho, theta)
3D Cartesian (x, y, z)
3D Spherical Polar
3D Cylindrical Polar

In fact Lagrangian mechanics simplifies the multitude of axes to a generic set
of axis \vec{q}.  As we depend on 2D Cartesian geometry for output to the
screen, and or paper documents etcetera, we thus need to convert from our axes to
2D Cartesian coordinates.

So as not to rewrite all the plot methods for every axes, we need our Base Axes
class to do the conversion for us through an Axes API.

Secondly, we need this API to convert from screen coordinates back to our Axes
coordinates so as to facilitate user interaction.

Finally, we need to stay aware of the fact that that some coordinate systems
do not have a 1:1 mapping with screen coordinates, that we will specify extra
parameters determined at run time to control this for example four parameters
to control the extent of the 2D Cartesian screen domain; in 3d parameters to
control the rotation and zoom.

As well as Axes 3D, Basemap should also welcome this change, with an anticipated
structure of a base mapping class with a coordinate system in lat/lon
coordinates, but with different mapping projections available for the
conversion between the Axes coordinate system and the screen.

Towards an API
--------------
First we define our coordinate transformation functions:
axes_to_base(self, *q)
base_to_axes(self, x, y)

The term ``base`` could get replaced with ``screen`` but for now we keep it
simple to reflect another transformation from base coords to screen coords,
e.g. perhaps to differentiate between window and screen coords.

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


