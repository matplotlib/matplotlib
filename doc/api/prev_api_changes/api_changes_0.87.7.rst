

Changes for 0.87.7
==================

.. code-block:: text

    Completely reworked the annotations API because I found the old
    API cumbersome.  The new design is much more legible and easy to
    read.  See matplotlib.text.Annotation and
    examples/annotation_demo.py

    markeredgecolor and markerfacecolor cannot be configured in
    matplotlibrc any more. Instead, markers are generally colored
    automatically based on the color of the line, unless marker colors
    are explicitly set as kwargs - NN

    Changed default comment character for load to '#' - JDH

    math_parse_s_ft2font_svg from mathtext.py & mathtext2.py now returns
    width, height, svg_elements. svg_elements is an instance of Bunch (
    cmbook.py) and has the attributes svg_glyphs and svg_lines, which are both
    lists.

    Renderer.draw_arc now takes an additional parameter, rotation.
    It specifies to draw the artist rotated in degrees anti-
    clockwise.  It was added for rotated ellipses.

    Renamed Figure.set_figsize_inches to Figure.set_size_inches to
    better match the get method, Figure.get_size_inches.

    Removed the copy_bbox_transform from transforms.py; added
    shallowcopy methods to all transforms.  All transforms already
    had deepcopy methods.

    FigureManager.resize(width, height): resize the window
    specified in pixels

    barh: x and y args have been renamed to width and bottom
    respectively, and their order has been swapped to maintain
    a (position, value) order.

    bar and barh: now accept kwarg 'edgecolor'.

    bar and barh: The left, height, width and bottom args can
    now all be scalars or sequences; see docstring.

    barh: now defaults to edge aligned instead of center
    aligned bars

    bar, barh and hist: Added a keyword arg 'align' that
    controls between edge or center bar alignment.

    Collections: PolyCollection and LineCollection now accept
    vertices or segments either in the original form [(x,y),
    (x,y), ...] or as a 2D numerix array, with X as the first column
    and Y as the second. Contour and quiver output the numerix
    form.  The transforms methods Bbox.update() and
    Transformation.seq_xy_tups() now accept either form.

    Collections: LineCollection is now a ScalarMappable like
    PolyCollection, etc.

    Specifying a grayscale color as a float is deprecated; use
    a string instead, e.g., 0.75 -> '0.75'.

    Collections: initializers now accept any mpl color arg, or
    sequence of such args; previously only a sequence of rgba
    tuples was accepted.

    Colorbar: completely new version and api; see docstring.  The
    original version is still accessible as colorbar_classic, but
    is deprecated.

    Contourf: "extend" kwarg replaces "clip_ends"; see docstring.
    Masked array support added to pcolormesh.

    Modified aspect-ratio handling:
        Removed aspect kwarg from imshow
        Axes methods:
            set_aspect(self, aspect, adjustable=None, anchor=None)
            set_adjustable(self, adjustable)
            set_anchor(self, anchor)
        Pylab interface:
            axis('image')

     Backend developers: ft2font's load_char now takes a flags
     argument, which you can OR together from the LOAD_XXX
     constants.
