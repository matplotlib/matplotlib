
Changes for 0.54
================

MATLAB interface
----------------

dpi
~~~

Several of the backends used a PIXELS_PER_INCH hack that I added to
try and make images render consistently across backends.  This just
complicated matters.  So you may find that some font sizes and line
widths appear different than before.  Apologies for the
inconvenience. You should set the dpi to an accurate value for your
screen to get true sizes.


pcolor and scatter
~~~~~~~~~~~~~~~~~~

There are two changes to the MATLAB interface API, both involving the
patch drawing commands.  For efficiency, pcolor and scatter have been
rewritten to use polygon collections, which are a new set of objects
from matplotlib.collections designed to enable efficient handling of
large collections of objects.  These new collections make it possible
to build large scatter plots or pcolor plots with no loops at the
python level, and are significantly faster than their predecessors.
The original pcolor and scatter functions are retained as
pcolor_classic and scatter_classic.

The return value from pcolor is a PolyCollection.  Most of the
properties that are available on rectangles or other patches are also
available on PolyCollections, e.g., you can say::

  c = scatter(blah, blah)
  c.set_linewidth(1.0)
  c.set_facecolor('r')
  c.set_alpha(0.5)

or::

  c = scatter(blah, blah)
  set(c, 'linewidth', 1.0, 'facecolor', 'r', 'alpha', 0.5)


Because the collection is a single object, you no longer need to loop
over the return value of scatter or pcolor to set properties for the
entire list.

If you want the different elements of a collection to vary on a
property, e.g., to have different line widths, see matplotlib.collections
for a discussion on how to set the properties as a sequence.

For scatter, the size argument is now in points^2 (the area of the
symbol in points) as in MATLAB and is not in data coords as before.
Using sizes in data coords caused several problems.  So you will need
to adjust your size arguments accordingly or use scatter_classic.

mathtext spacing
~~~~~~~~~~~~~~~~

For reasons not clear to me (and which I'll eventually fix) spacing no
longer works in font groups.  However, I added three new spacing
commands which compensate for this '\ ' (regular space), '\/' (small
space) and '\hspace{frac}' where frac is a fraction of fontsize in
points.  You will need to quote spaces in font strings, is::

  title(r'$\rm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')



Object interface - Application programmers
------------------------------------------

Autoscaling
~~~~~~~~~~~

The x and y axis instances no longer have autoscale view.  These are
handled by axes.autoscale_view

Axes creation
~~~~~~~~~~~~~

You should not instantiate your own Axes any more using the OO API.
Rather, create a Figure as before and in place of::

    f = Figure(figsize=(5,4), dpi=100)
    a = Subplot(f, 111)
    f.add_axis(a)

use::

    f = Figure(figsize=(5,4), dpi=100)
    a = f.add_subplot(111)

That is, add_axis no longer exists and is replaced by::

    add_axes(rect, axisbg=defaultcolor, frameon=True)
    add_subplot(num, axisbg=defaultcolor, frameon=True)

Artist methods
~~~~~~~~~~~~~~

If you define your own Artists, you need to rename the _draw method
to draw

Bounding boxes
~~~~~~~~~~~~~~

matplotlib.transforms.Bound2D is replaced by
matplotlib.transforms.Bbox.  If you want to construct a bbox from
left, bottom, width, height (the signature for Bound2D), use
matplotlib.transforms.lbwh_to_bbox, as in::

    bbox = clickBBox = lbwh_to_bbox(left, bottom, width, height)

The Bbox has a different API than the Bound2D.  e.g., if you want to
get the width and height of the bbox

**OLD**::

     width  = fig.bbox.x.interval()
     height = fig.bbox.y.interval()

**NEW**::

     width  = fig.bbox.width()
     height = fig.bbox.height()


Object constructors
~~~~~~~~~~~~~~~~~~~

You no longer pass the bbox, dpi, or transforms to the various
Artist constructors.  The old way or creating lines and rectangles
was cumbersome because you had to pass so many attributes to the
Line2D and Rectangle classes not related directly to the geometry
and properties of the object.  Now default values are added to the
object when you call axes.add_line or axes.add_patch, so they are
hidden from the user.

If you want to define a custom transformation on these objects, call
o.set_transform(trans) where trans is a Transformation instance.

In prior versions of you wanted to add a custom line in data coords,
you would have to do::

    l = Line2D(dpi, bbox, x, y,
               color = color,
               transx = transx,
               transy = transy,
               )

now all you need is::

    l = Line2D(x, y, color=color)

and the axes will set the transformation for you (unless you have
set your own already, in which case it will eave it unchanged)

Transformations
~~~~~~~~~~~~~~~

The entire transformation architecture has been rewritten.
Previously the x and y transformations where stored in the xaxis and
yaxis instances.  The problem with this approach is it only allows
for separable transforms (where the x and y transformations don't
depend on one another).  But for cases like polar, they do.  Now
transformations operate on x,y together.  There is a new base class
matplotlib.transforms.Transformation and two concrete
implementations, matplotlib.transforms.SeparableTransformation and
matplotlib.transforms.Affine.  The SeparableTransformation is
constructed with the bounding box of the input (this determines the
rectangular coordinate system of the input, i.e., the x and y view
limits), the bounding box of the display, and possibly nonlinear
transformations of x and y.  The 2 most frequently used
transformations, data coordinates -> display and axes coordinates ->
display are available as ax.transData and ax.transAxes.  See
alignment_demo.py which uses axes coords.

Also, the transformations should be much faster now, for two reasons

* they are written entirely in extension code

* because they operate on x and y together, they can do the entire
  transformation in one loop.  Earlier I did something along the
  lines of::

    xt = sx*func(x) + tx
    yt = sy*func(y) + ty

  Although this was done in numerix, it still involves 6 length(x)
  for-loops (the multiply, add, and function evaluation each for x
  and y).  Now all of that is done in a single pass.

If you are using transformations and bounding boxes to get the
cursor position in data coordinates, the method calls are a little
different now.  See the updated examples/coords_demo.py which shows
you how to do this.

Likewise, if you are using the artist bounding boxes to pick items
on the canvas with the GUI, the bbox methods are somewhat
different.  You will need to see the updated
examples/object_picker.py.

See unit/transforms_unit.py for many examples using the new
transformations.


.. highlight:: none
