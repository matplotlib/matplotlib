.. _toolkit_mplot3d-faq:

***********
mplot3d FAQ
***********

How is mplot3d different from MayaVi?
=====================================
`MayaVi2 <http://code.enthought.com/projects/mayavi/documentation.php>`_
is a very powerful and featureful 3D graphing library. For advanced
3D scenes and excellent rendering capabilities, it is highly recomended to
use MayaVi2.

mplot3d was intended to allow users to create simple 3D graphs with the same
"look-and-feel" as matplotlib's 2D plots. Furthermore, users can use the same
toolkit that they are already familiar with to generate both their 2D and 3D
plots.


My 3D plot doesn't look right at certain viewing angles
=======================================================
This is probably the most commonly reported issue with mplot3d. The problem
is that -- from some viewing angles -- a 3D object would appear in front
of another object, even though it is physically behind it. This can result in
plots that do not look "physically correct."

Unfortunately, while some work is being done to reduce the occurance of this
artifact, it is currently an intractable problem, and can not be fully solved
until matplotlib supports 3D graphics rendering at its core.

The problem occurs due to the reduction of 3D data down to 2D + z-order
scalar. A single value represents the 3rd dimension for all parts of 3D
objects in a collection. Therefore, when the bounding boxes of two collections
intersect, it becomes possible for this artifact to occur. Furthermore, the
intersection of two 3D objects (such as polygons or patches) can not be
rendered properly in matplotlib's 2D rendering engine.

This problem will likely not be solved until OpenGL support is added to all of
the backends (patches are greatly welcomed). Until then, if you need complex
3D scenes, we recommend using
`MayaVi <http://code.enthought.com/projects/mayavi/documentation.php>`_.


I don't like how the 3D plot is laid out, how do I change that?
===============================================================
Historically, mplot3d has suffered from a hard-coding of parameters used
to control visuals such as label spacing, tick length, and grid line width.
Work is being done to eliminate this issue. For matplotlib v1.1.0, there is
a semi-official manner to modify these parameters. See the note in the
:ref:`toolkit_mplot3d-axisapi` section of the mplot3d API documentation for
more information.

