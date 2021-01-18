import contextlib
import functools
import inspect
import math
from numbers import Number
import textwrap
from collections import namedtuple

import numpy as np

import matplotlib as mpl
from . import (_api, artist, cbook, colors, docstring, hatch as mhatch,
               lines as mlines, transforms)
from .bezier import (
    NonIntersectingPathException, get_cos_sin, get_intersection,
    get_parallels, inside_circle, make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath, split_path_inout)
from .path import Path


@cbook._define_aliases({
    "antialiased": ["aa"],
    "edgecolor": ["ec"],
    "facecolor": ["fc"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
})
class Patch(artist.Artist):
    """
    A patch is a 2D artist with a face color and an edge color.

    If any of *edgecolor*, *facecolor*, *linewidth*, or *antialiased*
    are *None*, they default to their rc params setting.
    """
    zorder = 1

    @_api.deprecated("3.4")
    @_api.classproperty
    def validCap(cls):
        with _api.suppress_matplotlib_deprecation_warning():
            return mlines.Line2D.validCap

    @_api.deprecated("3.4")
    @_api.classproperty
    def validJoin(cls):
        with _api.suppress_matplotlib_deprecation_warning():
            return mlines.Line2D.validJoin

    # Whether to draw an edge by default.  Set on a
    # subclass-by-subclass basis.
    _edge_default = False

    def __init__(self,
                 edgecolor=None,
                 facecolor=None,
                 color=None,
                 linewidth=None,
                 linestyle=None,
                 antialiased=None,
                 hatch=None,
                 fill=True,
                 capstyle=None,
                 joinstyle=None,
                 **kwargs):
        """
        The following kwarg properties are supported

        %(Patch_kwdoc)s
        """
        super().__init__()

        if linewidth is None:
            linewidth = mpl.rcParams['patch.linewidth']
        if linestyle is None:
            linestyle = "solid"
        if capstyle is None:
            capstyle = 'butt'
        if joinstyle is None:
            joinstyle = 'miter'
        if antialiased is None:
            antialiased = mpl.rcParams['patch.antialiased']

        self._hatch_color = colors.to_rgba(mpl.rcParams['hatch.color'])
        self._fill = True  # needed for set_facecolor call
        if color is not None:
            if edgecolor is not None or facecolor is not None:
                _api.warn_external(
                    "Setting the 'color' property will override "
                    "the edgecolor or facecolor properties.")
            self.set_color(color)
        else:
            self.set_edgecolor(edgecolor)
            self.set_facecolor(facecolor)
        # unscaled dashes.  Needed to scale dash patterns by lw
        self._us_dashes = None
        self._linewidth = 0

        self.set_fill(fill)
        self.set_linestyle(linestyle)
        self.set_linewidth(linewidth)
        self.set_antialiased(antialiased)
        self.set_hatch(hatch)
        self.set_capstyle(capstyle)
        self.set_joinstyle(joinstyle)

        if len(kwargs):
            self.update(kwargs)

    def get_verts(self):
        """
        Return a copy of the vertices used in this patch.

        If the patch contains Bezier curves, the curves will be interpolated by
        line segments.  To access the curves as curves, use `get_path`.
        """
        trans = self.get_transform()
        path = self.get_path()
        polygons = path.to_polygons(trans)
        if len(polygons):
            return polygons[0]
        return []

    def _process_radius(self, radius):
        if radius is not None:
            return radius
        if isinstance(self._picker, Number):
            _radius = self._picker
        else:
            if self.get_edgecolor()[3] == 0:
                _radius = 0
            else:
                _radius = self.get_linewidth()
        return _radius

    def contains(self, mouseevent, radius=None):
        """
        Test whether the mouse event occurred in the patch.

        Returns
        -------
        (bool, empty dict)
        """
        inside, info = self._default_contains(mouseevent)
        if inside is not None:
            return inside, info
        radius = self._process_radius(radius)
        codes = self.get_path().codes
        if codes is not None:
            vertices = self.get_path().vertices
            # if the current path is concatenated by multiple sub paths.
            # get the indexes of the starting code(MOVETO) of all sub paths
            idxs, = np.where(codes == Path.MOVETO)
            # Don't split before the first MOVETO.
            idxs = idxs[1:]
            subpaths = map(
                Path, np.split(vertices, idxs), np.split(codes, idxs))
        else:
            subpaths = [self.get_path()]
        inside = any(
            subpath.contains_point(
                (mouseevent.x, mouseevent.y), self.get_transform(), radius)
            for subpath in subpaths)
        return inside, {}

    def contains_point(self, point, radius=None):
        """
        Return whether the given point is inside the patch.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check, in target coordinates of
            ``self.get_transform()``. These are display coordinates for patches
            that are added to a figure or axes.
        radius : float, optional
            Add an additional margin on the patch in target coordinates of
            ``self.get_transform()``. See `.Path.contains_point` for further
            details.

        Returns
        -------
        bool

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        Isolated patches do not have a transform. In this case, the patch
        creation coordinates and the point coordinates match. The following
        example checks that the center of a circle is within the circle

        >>> center = 0, 0
        >>> c = Circle(center, radius=1)
        >>> c.contains_point(center)
        True

        The convention of checking against the transformed patch stems from
        the fact that this method is predominantly used to check if display
        coordinates (e.g. from mouse events) are within the patch. If you want
        to do the above check with data coordinates, you have to properly
        transform them first:

        >>> center = 0, 0
        >>> c = Circle(center, radius=1)
        >>> plt.gca().add_patch(c)
        >>> transformed_center = c.get_transform().transform(center)
        >>> c.contains_point(transformed_center)
        True

        """
        radius = self._process_radius(radius)
        return self.get_path().contains_point(point,
                                              self.get_transform(),
                                              radius)

    def contains_points(self, points, radius=None):
        """
        Return whether the given points are inside the patch.

        Parameters
        ----------
        points : (N, 2) array
            The points to check, in target coordinates of
            ``self.get_transform()``. These are display coordinates for patches
            that are added to a figure or axes. Columns contain x and y values.
        radius : float, optional
            Add an additional margin on the patch in target coordinates of
            ``self.get_transform()``. See `.Path.contains_point` for further
            details.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        See the notes on `.Patch.contains_point`.
        """
        radius = self._process_radius(radius)
        return self.get_path().contains_points(points,
                                               self.get_transform(),
                                               radius)

    def update_from(self, other):
        # docstring inherited.
        super().update_from(other)
        # For some properties we don't need or don't want to go through the
        # getters/setters, so we just copy them directly.
        self._edgecolor = other._edgecolor
        self._facecolor = other._facecolor
        self._original_edgecolor = other._original_edgecolor
        self._original_facecolor = other._original_facecolor
        self._fill = other._fill
        self._hatch = other._hatch
        self._hatch_color = other._hatch_color
        # copy the unscaled dash pattern
        self._us_dashes = other._us_dashes
        self.set_linewidth(other._linewidth)  # also sets dash properties
        self.set_transform(other.get_data_transform())
        # If the transform of other needs further initialization, then it will
        # be the case for this artist too.
        self._transformSet = other.is_transform_set()

    def get_extents(self):
        """
        Return the `Patch`'s axis-aligned extents as a `~.transforms.Bbox`.
        """
        return self.get_path().get_extents(self.get_transform())

    def get_transform(self):
        """Return the `~.transforms.Transform` applied to the `Patch`."""
        return self.get_patch_transform() + artist.Artist.get_transform(self)

    def get_data_transform(self):
        """
        Return the `~.transforms.Transform` mapping data coordinates to
        physical coordinates.
        """
        return artist.Artist.get_transform(self)

    def get_patch_transform(self):
        """
        Return the `~.transforms.Transform` instance mapping patch coordinates
        to data coordinates.

        For example, one may define a patch of a circle which represents a
        radius of 5 by providing coordinates for a unit circle, and a
        transform which scales the coordinates (the patch coordinate) by 5.
        """
        return transforms.IdentityTransform()

    def get_antialiased(self):
        """Return whether antialiasing is used for drawing."""
        return self._antialiased

    def get_edgecolor(self):
        """Return the edge color."""
        return self._edgecolor

    def get_facecolor(self):
        """Return the face color."""
        return self._facecolor

    def get_linewidth(self):
        """Return the line width in points."""
        return self._linewidth

    def get_linestyle(self):
        """Return the linestyle."""
        return self._linestyle

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        b : bool or None
        """
        if aa is None:
            aa = mpl.rcParams['patch.antialiased']
        self._antialiased = aa
        self.stale = True

    def _set_edgecolor(self, color):
        set_hatch_color = True
        if color is None:
            if (mpl.rcParams['patch.force_edgecolor'] or
                    not self._fill or self._edge_default):
                color = mpl.rcParams['patch.edgecolor']
            else:
                color = 'none'
                set_hatch_color = False

        self._edgecolor = colors.to_rgba(color, self._alpha)
        if set_hatch_color:
            self._hatch_color = self._edgecolor
        self.stale = True

    def set_edgecolor(self, color):
        """
        Set the patch edge color.

        Parameters
        ----------
        color : color or None or 'auto'
        """
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def _set_facecolor(self, color):
        if color is None:
            color = mpl.rcParams['patch.facecolor']
        alpha = self._alpha if self._fill else 0
        self._facecolor = colors.to_rgba(color, alpha)
        self.stale = True

    def set_facecolor(self, color):
        """
        Set the patch face color.

        Parameters
        ----------
        color : color or None
        """
        self._original_facecolor = color
        self._set_facecolor(color)

    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.

        Parameters
        ----------
        c : color

        See Also
        --------
        Patch.set_facecolor, Patch.set_edgecolor
            For setting the edge or face color individually.
        """
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_alpha(self, alpha):
        # docstring inherited
        super().set_alpha(alpha)
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        # stale is already True

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points.

        Parameters
        ----------
        w : float or None
        """
        if w is None:
            w = mpl.rcParams['patch.linewidth']
            if w is None:
                w = mpl.rcParams['axes.linewidth']

        self._linewidth = float(w)
        # scale the dash pattern by the linewidth
        offset, ls = self._us_dashes
        self._dashoffset, self._dashes = mlines._scale_dashes(
            offset, ls, self._linewidth)
        self.stale = True

    def set_linestyle(self, ls):
        """
        Set the patch linestyle.

        ===========================   =================
        linestyle                     description
        ===========================   =================
        ``'-'`` or ``'solid'``        solid line
        ``'--'`` or  ``'dashed'``     dashed line
        ``'-.'`` or  ``'dashdot'``    dash-dotted line
        ``':'`` or ``'dotted'``       dotted line
        ``'None'``                    draw nothing
        ``'none'``                    draw nothing
        ``' '``                       draw nothing
        ``''``                        draw nothing
        ===========================   =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq)

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            The line style.
        """
        if ls is None:
            ls = "solid"
        if ls in [' ', '', 'none']:
            ls = 'None'
        self._linestyle = ls
        # get the unscaled dash pattern
        offset, ls = self._us_dashes = mlines._get_dash_pattern(ls)
        # scale the dash pattern by the linewidth
        self._dashoffset, self._dashes = mlines._scale_dashes(
            offset, ls, self._linewidth)
        self.stale = True

    def set_fill(self, b):
        """
        Set whether to fill the patch.

        Parameters
        ----------
        b : bool
        """
        self._fill = bool(b)
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        self.stale = True

    def get_fill(self):
        """Return whether the patch is filled."""
        return self._fill

    # Make fill a property so as to preserve the long-standing
    # but somewhat inconsistent behavior in which fill was an
    # attribute.
    fill = property(get_fill, set_fill)

    def set_capstyle(self, s):
        """
        Set the capstyle.

        Parameters
        ----------
        s : {'butt', 'round', 'projecting'}
        """
        mpl.rcsetup.validate_capstyle(s)
        self._capstyle = s
        self.stale = True

    def get_capstyle(self):
        """Return the capstyle."""
        return self._capstyle

    def set_joinstyle(self, s):
        """
        Set the joinstyle.

        Parameters
        ----------
        s : {'miter', 'round', 'bevel'}
        """
        mpl.rcsetup.validate_joinstyle(s)
        self._joinstyle = s
        self.stale = True

    def get_joinstyle(self):
        """Return the joinstyle."""
        return self._joinstyle

    def set_hatch(self, hatch):
        r"""
        Set the hatching pattern.

        *hatch* can be one of::

          /   - diagonal hatching
          \   - back diagonal
          |   - vertical
          -   - horizontal
          +   - crossed
          x   - crossed diagonal
          o   - small circle
          O   - large circle
          .   - dots
          *   - stars

        Letters can be combined, in which case all the specified
        hatchings are done.  If same letter repeats, it increases the
        density of hatching of that pattern.

        Hatching is supported in the PostScript, PDF, SVG and Agg
        backends only.

        Parameters
        ----------
        hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        """
        # Use validate_hatch(list) after deprecation.
        mhatch._validate_hatch_pattern(hatch)
        self._hatch = hatch
        self.stale = True

    def get_hatch(self):
        """Return the hatching pattern."""
        return self._hatch

    @contextlib.contextmanager
    def _bind_draw_path_function(self, renderer):
        """
        ``draw()`` helper factored out for sharing with `FancyArrowPatch`.

        Yields a callable ``dp`` such that calling ``dp(*args, **kwargs)`` is
        equivalent to calling ``renderer1.draw_path(gc, *args, **kwargs)``
        where ``renderer1`` and ``gc`` have been suitably set from ``renderer``
        and the artist's properties.
        """

        renderer.open_group('patch', self.get_gid())
        gc = renderer.new_gc()

        gc.set_foreground(self._edgecolor, isRGBA=True)

        lw = self._linewidth
        if self._edgecolor[3] == 0 or self._linestyle == 'None':
            lw = 0
        gc.set_linewidth(lw)
        gc.set_dashes(self._dashoffset, self._dashes)
        gc.set_capstyle(self._capstyle)
        gc.set_joinstyle(self._joinstyle)

        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_url(self._url)
        gc.set_snap(self.get_snap())

        gc.set_alpha(self._alpha)

        if self._hatch:
            gc.set_hatch(self._hatch)
            gc.set_hatch_color(self._hatch_color)

        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        # In `with _bind_draw_path_function(renderer) as draw_path: ...`
        # (in the implementations of `draw()` below), calls to `draw_path(...)`
        # will occur as if they took place here with `gc` inserted as
        # additional first argument.
        yield functools.partial(renderer.draw_path, gc)

        gc.restore()
        renderer.close_group('patch')
        self.stale = False

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return
        # Patch has traditionally ignored the dashoffset.
        with cbook._setattr_cm(self, _dashoffset=0), \
                self._bind_draw_path_function(renderer) as draw_path:
            path = self.get_path()
            transform = self.get_transform()
            tpath = transform.transform_path_non_affine(path)
            affine = transform.get_affine()
            draw_path(tpath, affine,
                      # Work around a bug in the PDF and SVG renderers, which
                      # do not draw the hatches if the facecolor is fully
                      # transparent, but do if it is None.
                      self._facecolor if self._facecolor[3] else None)

    def get_path(self):
        """Return the path of this patch."""
        raise NotImplementedError('Derived must override')

    def get_window_extent(self, renderer=None):
        return self.get_path().get_extents(self.get_transform())

    def _convert_xy_units(self, xy):
        """Convert x and y units for a tuple (x, y)."""
        x = self.convert_xunits(xy[0])
        y = self.convert_yunits(xy[1])
        return x, y


_patch_kwdoc = artist.kwdoc(Patch)
for k in ['Rectangle', 'Circle', 'RegularPolygon', 'Polygon', 'Wedge', 'Arrow',
          'FancyArrow', 'CirclePolygon', 'Ellipse', 'Arc', 'FancyBboxPatch',
          'Patch']:
    docstring.interpd.update({f'{k}_kwdoc': _patch_kwdoc})

# define Patch.__init__ docstring after the class has been added to interpd
docstring.dedent_interpd(Patch.__init__)


class Shadow(Patch):
    def __str__(self):
        return "Shadow(%s)" % (str(self.patch))

    @_api.delete_parameter("3.3", "props")
    @docstring.dedent_interpd
    def __init__(self, patch, ox, oy, props=None, **kwargs):
        """
        Create a shadow of the given *patch*.

        By default, the shadow will have the same face color as the *patch*,
        but darkened.

        Parameters
        ----------
        patch : `.Patch`
            The patch to create the shadow for.
        ox, oy : float
            The shift of the shadow in data coordinates, scaled by a factor
            of dpi/72.
        props : dict
            *deprecated (use kwargs instead)* Properties of the shadow patch.
        **kwargs
            Properties of the shadow patch. Supported keys are:

            %(Patch_kwdoc)s
        """
        super().__init__()
        self.patch = patch
        # Note: when removing props, we can directly pass kwargs to _update()
        # and remove self._props
        if props is None:
            color = .3 * np.asarray(colors.to_rgb(self.patch.get_facecolor()))
            props = {
                'facecolor': color,
                'edgecolor': color,
                'alpha': 0.5,
            }
        self._props = {**props, **kwargs}
        self._ox, self._oy = ox, oy
        self._shadow_transform = transforms.Affine2D()
        self._update()

    props = _api.deprecate_privatize_attribute("3.3")

    def _update(self):
        self.update_from(self.patch)

        # Place the shadow patch directly behind the inherited patch.
        self.set_zorder(np.nextafter(self.patch.zorder, -np.inf))

        self.update(self._props)

    def _update_transform(self, renderer):
        ox = renderer.points_to_pixels(self._ox)
        oy = renderer.points_to_pixels(self._oy)
        self._shadow_transform.clear().translate(ox, oy)

    def get_path(self):
        return self.patch.get_path()

    def get_patch_transform(self):
        return self.patch.get_patch_transform() + self._shadow_transform

    def draw(self, renderer):
        self._update_transform(renderer)
        super().draw(renderer)


class Rectangle(Patch):
    """
    A rectangle defined via an anchor point *xy* and its *width* and *height*.

    The rectangle extends from ``xy[0]`` to ``xy[0] + width`` in x-direction
    and from ``xy[1]`` to ``xy[1] + height`` in y-direction. ::

      :                +------------------+
      :                |                  |
      :              height               |
      :                |                  |
      :               (xy)---- width -----+

    One may picture *xy* as the bottom left corner, but which corner *xy* is
    actually depends on the the direction of the axis and the sign of *width*
    and *height*; e.g. *xy* would be the bottom right corner if the x-axis
    was inverted or if *width* was negative.
    """

    def __str__(self):
        pars = self._x0, self._y0, self._width, self._height, self.angle
        fmt = "Rectangle(xy=(%g, %g), width=%g, height=%g, angle=%g)"
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, xy, width, height, angle=0.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            %(Patch_kwdoc)s
        """
        super().__init__(**kwargs)
        self._x0 = xy[0]
        self._y0 = xy[1]
        self._width = width
        self._height = height
        self.angle = float(angle)
        self._convert_units()  # Validate the inputs.

    def get_path(self):
        """Return the vertices of the rectangle."""
        return Path.unit_rectangle()

    def _convert_units(self):
        """Convert bounds of the rectangle."""
        x0 = self.convert_xunits(self._x0)
        y0 = self.convert_yunits(self._y0)
        x1 = self.convert_xunits(self._x0 + self._width)
        y1 = self.convert_yunits(self._y0 + self._height)
        return x0, y0, x1, y1

    def get_patch_transform(self):
        # Note: This cannot be called until after this has been added to
        # an Axes, otherwise unit conversion will fail. This makes it very
        # important to call the accessor method and not directly access the
        # transformation member variable.
        bbox = self.get_bbox()
        return (transforms.BboxTransformTo(bbox)
                + transforms.Affine2D().rotate_deg_around(
                    bbox.x0, bbox.y0, self.angle))

    def get_x(self):
        """Return the left coordinate of the rectangle."""
        return self._x0

    def get_y(self):
        """Return the bottom coordinate of the rectangle."""
        return self._y0

    def get_xy(self):
        """Return the left and bottom coords of the rectangle as a tuple."""
        return self._x0, self._y0

    def get_width(self):
        """Return the width of the rectangle."""
        return self._width

    def get_height(self):
        """Return the height of the rectangle."""
        return self._height

    def set_x(self, x):
        """Set the left coordinate of the rectangle."""
        self._x0 = x
        self.stale = True

    def set_y(self, y):
        """Set the bottom coordinate of the rectangle."""
        self._y0 = y
        self.stale = True

    def set_xy(self, xy):
        """
        Set the left and bottom coordinates of the rectangle.

        Parameters
        ----------
        xy : (float, float)
        """
        self._x0, self._y0 = xy
        self.stale = True

    def set_width(self, w):
        """Set the width of the rectangle."""
        self._width = w
        self.stale = True

    def set_height(self, h):
        """Set the height of the rectangle."""
        self._height = h
        self.stale = True

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle as *left*, *bottom*, *width*, *height*.

        The values may be passed as separate parameters or as a tuple::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        .. ACCEPTS: (left, bottom, width, height)
        """
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        self._x0 = l
        self._y0 = b
        self._width = w
        self._height = h
        self.stale = True

    def get_bbox(self):
        """Return the `.Bbox`."""
        x0, y0, x1, y1 = self._convert_units()
        return transforms.Bbox.from_extents(x0, y0, x1, y1)

    xy = property(get_xy, set_xy)


class RegularPolygon(Patch):
    """A regular polygon patch."""

    def __str__(self):
        s = "RegularPolygon((%g, %g), %d, radius=%g, orientation=%g)"
        return s % (self.xy[0], self.xy[1], self.numvertices, self.radius,
                    self.orientation)

    @docstring.dedent_interpd
    def __init__(self, xy, numVertices, radius=5, orientation=0,
                 **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The center position.

        numVertices : int
            The number of vertices.

        radius : float
            The distance from the center to each of the vertices.

        orientation : float
            The polygon rotation angle (in radians).

        **kwargs
            `Patch` properties:

            %(Patch_kwdoc)s
        """
        self.xy = xy
        self.numvertices = numVertices
        self.orientation = orientation
        self.radius = radius
        self._path = Path.unit_regular_polygon(numVertices)
        self._patch_transform = transforms.Affine2D()
        super().__init__(**kwargs)

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return self._patch_transform.clear() \
            .scale(self.radius) \
            .rotate(self.orientation) \
            .translate(*self.xy)


class PathPatch(Patch):
    """A general polycurve path patch."""

    _edge_default = True

    def __str__(self):
        s = "PathPatch%d((%g, %g) ...)"
        return s % (len(self._path.vertices), *tuple(self._path.vertices[0]))

    @docstring.dedent_interpd
    def __init__(self, path, **kwargs):
        """
        *path* is a `~.path.Path` object.

        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(**kwargs)
        self._path = path

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path


class StepPatch(PathPatch):
    """
    A path patch describing a stepwise constant function.

    By default the path is not closed and starts and stops at
    baseline value.
    """

    _edge_default = False

    @docstring.dedent_interpd
    def __init__(self, values, edges, *,
                 orientation='vertical', baseline=0, **kwargs):
        """
        Parameters
        ----------
        values : array-like
            The step heights.

        edges : array-like
            The edge positions, with ``len(edges) == len(vals) + 1``,
            between which the curve takes on vals values.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            The direction of the steps. Vertical means that *values* are
            along the y-axis, and edges are along the x-axis.

        baseline : float, array-like or None, default: 0
            The bottom value of the bounding edges or when
            ``fill=True``, position of lower edge. If *fill* is
            True or an array is passed to *baseline*, a closed
            path is drawn.

        Other valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        self.orientation = orientation
        self._edges = np.asarray(edges)
        self._values = np.asarray(values)
        self._baseline = np.asarray(baseline) if baseline is not None else None
        self._update_path()
        super().__init__(self._path, **kwargs)

    def _update_path(self):
        if np.isnan(np.sum(self._edges)):
            raise ValueError('Nan values in "edges" are disallowed')
        if self._edges.size - 1 != self._values.size:
            raise ValueError('Size mismatch between "values" and "edges". '
                             "Expected `len(values) + 1 == len(edges)`, but "
                             f"`len(values) = {self._values.size}` and "
                             f"`len(edges) = {self._edges.size}`.")
        verts, codes = [], []

        _nan_mask = np.isnan(self._values)
        if self._baseline is not None:
            _nan_mask |= np.isnan(self._baseline)
        for idx0, idx1 in cbook.contiguous_regions(~_nan_mask):
            x = np.repeat(self._edges[idx0:idx1+1], 2)
            y = np.repeat(self._values[idx0:idx1], 2)
            if self._baseline is None:
                y = np.hstack((y[0], y, y[-1]))
            elif self._baseline.ndim == 0:  # single baseline value
                y = np.hstack((self._baseline, y, self._baseline))
            elif self._baseline.ndim == 1:  # baseline array
                base = np.repeat(self._baseline[idx0:idx1], 2)[::-1]
                x = np.concatenate([x, x[::-1]])
                y = np.concatenate([np.hstack((base[-1], y, base[0],
                                               base[0], base, base[-1]))])
            else:  # no baseline
                raise ValueError('Invalid `baseline` specified')
            if self.orientation == 'vertical':
                xy = np.column_stack([x, y])
            else:
                xy = np.column_stack([y, x])
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO]*(len(xy)-1))
        self._path = Path(np.concatenate(verts), np.concatenate(codes))

    def get_data(self):
        """Get `.StepPatch` values, edges and baseline as namedtuple."""
        StairData = namedtuple('StairData', 'values edges baseline')
        return StairData(self._values, self._edges, self._baseline)

    def set_data(self, values=None, edges=None, baseline=None):
        """
        Set `.StepPatch` values, edges and baseline.

        Parameters
        ----------
        values : 1D array-like or None
            Will not update values, if passing None
        edges : 1D array-like, optional
        baseline : float, 1D array-like or None
        """
        if values is None and edges is None and baseline is None:
            raise ValueError("Must set *values*, *edges* or *baseline*.")
        if values is not None:
            self._values = np.asarray(values)
        if edges is not None:
            self._edges = np.asarray(edges)
        if baseline is not None:
            self._baseline = np.asarray(baseline)
        self._update_path()
        self.stale = True


class Polygon(Patch):
    """A general polygon patch."""

    def __str__(self):
        s = "Polygon%d((%g, %g) ...)"
        return s % (len(self._path.vertices), *tuple(self._path.vertices[0]))

    @docstring.dedent_interpd
    def __init__(self, xy, closed=True, **kwargs):
        """
        *xy* is a numpy array with shape Nx2.

        If *closed* is *True*, the polygon will be closed so the
        starting and ending points are the same.

        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(**kwargs)
        self._closed = closed
        self.set_xy(xy)

    def get_path(self):
        """Get the `.Path` of the polygon."""
        return self._path

    def get_closed(self):
        """Return whether the polygon is closed."""
        return self._closed

    def set_closed(self, closed):
        """
        Set whether the polygon is closed.

        Parameters
        ----------
        closed : bool
           True if the polygon is closed
        """
        if self._closed == bool(closed):
            return
        self._closed = bool(closed)
        self.set_xy(self.get_xy())
        self.stale = True

    def get_xy(self):
        """
        Get the vertices of the path.

        Returns
        -------
        (N, 2) numpy array
            The coordinates of the vertices.
        """
        return self._path.vertices

    def set_xy(self, xy):
        """
        Set the vertices of the polygon.

        Parameters
        ----------
        xy : (N, 2) array-like
            The coordinates of the vertices.

        Notes
        -----
        Unlike `~.path.Path`, we do not ignore the last input vertex. If the
        polygon is meant to be closed, and the last point of the polygon is not
        equal to the first, we assume that the user has not explicitly passed a
        ``CLOSEPOLY`` vertex, and add it ourselves.
        """
        xy = np.asarray(xy)
        nverts, _ = xy.shape
        if self._closed:
            # if the first and last vertex are the "same", then we assume that
            # the user explicitly passed the CLOSEPOLY vertex. Otherwise, we
            # have to append one since the last vertex will be "ignored" by
            # Path
            if nverts == 1 or nverts > 1 and (xy[0] != xy[-1]).any():
                xy = np.concatenate([xy, [xy[0]]])
        else:
            # if we aren't closed, and the last vertex matches the first, then
            # we assume we have an unnecessary CLOSEPOLY vertex and remove it
            if nverts > 2 and (xy[0] == xy[-1]).all():
                xy = xy[:-1]
        self._path = Path(xy, closed=self._closed)
        self.stale = True

    xy = property(get_xy, set_xy,
                  doc='The vertices of the path as (N, 2) numpy array.')


class Wedge(Patch):
    """Wedge shaped patch."""

    def __str__(self):
        pars = (self.center[0], self.center[1], self.r,
                self.theta1, self.theta2, self.width)
        fmt = "Wedge(center=(%g, %g), r=%g, theta1=%g, theta2=%g, width=%s)"
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, center, r, theta1, theta2, width=None, **kwargs):
        """
        A wedge centered at *x*, *y* center with radius *r* that
        sweeps *theta1* to *theta2* (in degrees).  If *width* is given,
        then a partial wedge is drawn from inner radius *r* - *width*
        to outer radius *r*.

        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(**kwargs)
        self.center = center
        self.r, self.width = r, width
        self.theta1, self.theta2 = theta1, theta2
        self._patch_transform = transforms.IdentityTransform()
        self._recompute_path()

    def _recompute_path(self):
        # Inner and outer rings are connected unless the annulus is complete
        if abs((self.theta2 - self.theta1) - 360) <= 1e-12:
            theta1, theta2 = 0, 360
            connector = Path.MOVETO
        else:
            theta1, theta2 = self.theta1, self.theta2
            connector = Path.LINETO

        # Form the outer ring
        arc = Path.arc(theta1, theta2)

        if self.width is not None:
            # Partial annulus needs to draw the outer ring
            # followed by a reversed and scaled inner ring
            v1 = arc.vertices
            v2 = arc.vertices[::-1] * (self.r - self.width) / self.r
            v = np.vstack([v1, v2, v1[0, :], (0, 0)])
            c = np.hstack([arc.codes, arc.codes, connector, Path.CLOSEPOLY])
            c[len(arc.codes)] = connector
        else:
            # Wedge doesn't need an inner ring
            v = np.vstack([arc.vertices, [(0, 0), arc.vertices[0, :], (0, 0)]])
            c = np.hstack([arc.codes, [connector, connector, Path.CLOSEPOLY]])

        # Shift and scale the wedge to the final location.
        v *= self.r
        v += np.asarray(self.center)
        self._path = Path(v, c)

    def set_center(self, center):
        self._path = None
        self.center = center
        self.stale = True

    def set_radius(self, radius):
        self._path = None
        self.r = radius
        self.stale = True

    def set_theta1(self, theta1):
        self._path = None
        self.theta1 = theta1
        self.stale = True

    def set_theta2(self, theta2):
        self._path = None
        self.theta2 = theta2
        self.stale = True

    def set_width(self, width):
        self._path = None
        self.width = width
        self.stale = True

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path


# COVERAGE NOTE: Not used internally or from examples
class Arrow(Patch):
    """An arrow patch."""

    def __str__(self):
        return "Arrow()"

    _path = Path([[0.0, 0.1], [0.0, -0.1],
                  [0.8, -0.1], [0.8, -0.3],
                  [1.0, 0.0], [0.8, 0.3],
                  [0.8, 0.1], [0.0, 0.1]],
                 closed=True)

    @docstring.dedent_interpd
    def __init__(self, x, y, dx, dy, width=1.0, **kwargs):
        """
        Draws an arrow from (*x*, *y*) to (*x* + *dx*, *y* + *dy*).
        The width of the arrow is scaled by *width*.

        Parameters
        ----------
        x : float
            x coordinate of the arrow tail.
        y : float
            y coordinate of the arrow tail.
        dx : float
            Arrow length in the x direction.
        dy : float
            Arrow length in the y direction.
        width : float, default: 1
            Scale factor for the width of the arrow. With a default value of 1,
            the tail width is 0.2 and head width is 0.6.
        **kwargs
            Keyword arguments control the `Patch` properties:

            %(Patch_kwdoc)s

        See Also
        --------
        FancyArrow
            Patch that allows independent control of the head and tail
            properties.
        """
        super().__init__(**kwargs)
        self._patch_transform = (
            transforms.Affine2D()
            .scale(np.hypot(dx, dy), width)
            .rotate(np.arctan2(dy, dx))
            .translate(x, y)
            .frozen())

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return self._patch_transform


class FancyArrow(Polygon):
    """
    Like Arrow, but lets you set head width and head height independently.
    """

    _edge_default = True

    def __str__(self):
        return "FancyArrow()"

    @docstring.dedent_interpd
    def __init__(self, x, y, dx, dy, width=0.001, length_includes_head=False,
                 head_width=None, head_length=None, shape='full', overhang=0,
                 head_starts_at_zero=False, **kwargs):
        """
        Parameters
        ----------
        width : float, default: 0.001
            Width of full arrow tail.

        length_includes_head : bool, default: False
            True if head is to be counted in calculating the length.

        head_width : float or None, default: 3*width
            Total width of the full arrow head.

        head_length : float or None, default: 1.5*head_width
            Length of arrow head.

        shape : {'full', 'left', 'right'}, default: 'full'
            Draw the left-half, right-half, or full arrow.

        overhang : float, default: 0
            Fraction that the arrow is swept back (0 overhang means
            triangular shape). Can be negative or greater than one.

        head_starts_at_zero : bool, default: False
            If True, the head starts being drawn at coordinate 0
            instead of ending at coordinate 0.

        **kwargs
            `.Patch` properties:

            %(Patch_kwdoc)s
        """
        if head_width is None:
            head_width = 3 * width
        if head_length is None:
            head_length = 1.5 * head_width

        distance = np.hypot(dx, dy)

        if length_includes_head:
            length = distance
        else:
            length = distance + head_length
        if not length:
            verts = np.empty([0, 2])  # display nothing if empty
        else:
            # start by drawing horizontal arrow, point at (0, 0)
            hw, hl, hs, lw = head_width, head_length, overhang, width
            left_half_arrow = np.array([
                [0.0, 0.0],                 # tip
                [-hl, -hw / 2],             # leftmost
                [-hl * (1 - hs), -lw / 2],  # meets stem
                [-length, -lw / 2],         # bottom left
                [-length, 0],
            ])
            # if we're not including the head, shift up by head length
            if not length_includes_head:
                left_half_arrow += [head_length, 0]
            # if the head starts at 0, shift up by another head length
            if head_starts_at_zero:
                left_half_arrow += [head_length / 2, 0]
            # figure out the shape, and complete accordingly
            if shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow * [1, -1]
                if shape == 'right':
                    coords = right_half_arrow
                elif shape == 'full':
                    # The half-arrows contain the midpoint of the stem,
                    # which we can omit from the full arrow. Including it
                    # twice caused a problem with xpdf.
                    coords = np.concatenate([left_half_arrow[:-1],
                                             right_half_arrow[-2::-1]])
                else:
                    raise ValueError("Got unknown shape: %s" % shape)
            if distance != 0:
                cx = dx / distance
                sx = dy / distance
            else:
                # Account for division by zero
                cx, sx = 0, 1
            M = [[cx, sx], [-sx, cx]]
            verts = np.dot(coords, M) + (x + dx, y + dy)

        super().__init__(verts, closed=True, **kwargs)


docstring.interpd.update(
    FancyArrow="\n".join(inspect.getdoc(FancyArrow.__init__).splitlines()[2:]))


class CirclePolygon(RegularPolygon):
    """A polygon-approximation of a circle patch."""

    def __str__(self):
        s = "CirclePolygon((%g, %g), radius=%g, resolution=%d)"
        return s % (self.xy[0], self.xy[1], self.radius, self.numvertices)

    @docstring.dedent_interpd
    def __init__(self, xy, radius=5,
                 resolution=20,  # the number of vertices
                 ** kwargs):
        """
        Create a circle at *xy* = (*x*, *y*) with given *radius*.

        This circle is approximated by a regular polygon with *resolution*
        sides.  For a smoother circle drawn with splines, see `Circle`.

        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(xy, resolution, radius, orientation=0, **kwargs)


class Ellipse(Patch):
    """A scale-free ellipse."""

    def __str__(self):
        pars = (self._center[0], self._center[1],
                self.width, self.height, self.angle)
        fmt = "Ellipse(xy=(%s, %s), width=%s, height=%s, angle=%s)"
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, xy, width, height, angle=0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of ellipse centre.
        width : float
            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        angle : float, default: 0
            Rotation in degrees anti-clockwise.

        Notes
        -----
        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(**kwargs)

        self._center = xy
        self._width, self._height = width, height
        self._angle = angle
        self._path = Path.unit_circle()
        # Note: This cannot be calculated until this is added to an Axes
        self._patch_transform = transforms.IdentityTransform()

    def _recompute_transform(self):
        """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
        center = (self.convert_xunits(self._center[0]),
                  self.convert_yunits(self._center[1]))
        width = self.convert_xunits(self._width)
        height = self.convert_yunits(self._height)
        self._patch_transform = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .rotate_deg(self.angle) \
            .translate(*center)

    def get_path(self):
        """Return the path of the ellipse."""
        return self._path

    def get_patch_transform(self):
        self._recompute_transform()
        return self._patch_transform

    def set_center(self, xy):
        """
        Set the center of the ellipse.

        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self.stale = True

    def get_center(self):
        """Return the center of the ellipse."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width of the ellipse.

        Parameters
        ----------
        width : float
        """
        self._width = width
        self.stale = True

    def get_width(self):
        """
        Return the width of the ellipse.
        """
        return self._width

    width = property(get_width, set_width)

    def set_height(self, height):
        """
        Set the height of the ellipse.

        Parameters
        ----------
        height : float
        """
        self._height = height
        self.stale = True

    def get_height(self):
        """Return the height of the ellipse."""
        return self._height

    height = property(get_height, set_height)

    def set_angle(self, angle):
        """
        Set the angle of the ellipse.

        Parameters
        ----------
        angle : float
        """
        self._angle = angle
        self.stale = True

    def get_angle(self):
        """Return the angle of the ellipse."""
        return self._angle

    angle = property(get_angle, set_angle)


class Circle(Ellipse):
    """A circle patch."""

    def __str__(self):
        pars = self.center[0], self.center[1], self.radius
        fmt = "Circle(xy=(%g, %g), radius=%g)"
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, xy, radius=5, **kwargs):
        """
        Create a true circle at center *xy* = (*x*, *y*) with given *radius*.

        Unlike `CirclePolygon` which is a polygonal approximation, this uses
        Bezier splines and is much closer to a scale-free circle.

        Valid keyword arguments are:

        %(Patch_kwdoc)s
        """
        super().__init__(xy, radius * 2, radius * 2, **kwargs)
        self.radius = radius

    def set_radius(self, radius):
        """
        Set the radius of the circle.

        Parameters
        ----------
        radius : float
        """
        self.width = self.height = 2 * radius
        self.stale = True

    def get_radius(self):
        """Return the radius of the circle."""
        return self.width / 2.

    radius = property(get_radius, set_radius)


class Arc(Ellipse):
    """
    An elliptical arc, i.e. a segment of an ellipse.

    Due to internal optimizations, there are certain restrictions on using Arc:

    - The arc cannot be filled.

    - The arc must be used in an `~.axes.Axes` instance. It can not be added
      directly to a `.Figure` because it is optimized to only render the
      segments that are inside the axes bounding box with high resolution.
    """
    def __str__(self):
        pars = (self.center[0], self.center[1], self.width,
                self.height, self.angle, self.theta1, self.theta2)
        fmt = ("Arc(xy=(%g, %g), width=%g, "
               "height=%g, angle=%g, theta1=%g, theta2=%g)")
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, xy, width, height, angle=0.0,
                 theta1=0.0, theta2=360.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The center of the ellipse.

        width : float
            The length of the horizontal axis.

        height : float
            The length of the vertical axis.

        angle : float
            Rotation of the ellipse in degrees (counterclockwise).

        theta1, theta2 : float, default: 0, 360
            Starting and ending angles of the arc in degrees. These values
            are relative to *angle*, e.g. if *angle* = 45 and *theta1* = 90
            the absolute starting angle is 135.
            Default *theta1* = 0, *theta2* = 360, i.e. a complete ellipse.
            The arc is drawn in the counterclockwise direction.
            Angles greater than or equal to 360, or smaller than 0, are
            represented by an equivalent angle in the range [0, 360), by
            taking the input value mod 360.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            Most `.Patch` properties are supported as keyword arguments,
            with the exception of *fill* and *facecolor* because filling is
            not supported.

        %(Patch_kwdoc)s
        """
        fill = kwargs.setdefault('fill', False)
        if fill:
            raise ValueError("Arc objects can not be filled")

        super().__init__(xy, width, height, angle, **kwargs)

        self.theta1 = theta1
        self.theta2 = theta2

    @artist.allow_rasterization
    def draw(self, renderer):
        """
        Draw the arc to the given *renderer*.

        Notes
        -----
        Ellipses are normally drawn using an approximation that uses
        eight cubic Bezier splines.  The error of this approximation
        is 1.89818e-6, according to this unverified source:

          Lancaster, Don.  *Approximating a Circle or an Ellipse Using
          Four Bezier Cubic Splines.*

          https://www.tinaja.com/glib/ellipse4.pdf

        There is a use case where very large ellipses must be drawn
        with very high accuracy, and it is too expensive to render the
        entire ellipse with enough segments (either splines or line
        segments).  Therefore, in the case where either radius of the
        ellipse is large enough that the error of the spline
        approximation will be visible (greater than one pixel offset
        from the ideal), a different technique is used.

        In that case, only the visible parts of the ellipse are drawn,
        with each visible arc using a fixed number of spline segments
        (8).  The algorithm proceeds as follows:

        1. The points where the ellipse intersects the axes bounding
           box are located.  (This is done be performing an inverse
           transformation on the axes bbox such that it is relative
           to the unit circle -- this makes the intersection
           calculation much easier than doing rotated ellipse
           intersection directly).

           This uses the "line intersecting a circle" algorithm from:

               Vince, John.  *Geometry for Computer Graphics: Formulae,
               Examples & Proofs.*  London: Springer-Verlag, 2005.

        2. The angles of each of the intersection points are calculated.

        3. Proceeding counterclockwise starting in the positive
           x-direction, each of the visible arc-segments between the
           pairs of vertices are drawn using the Bezier arc
           approximation technique implemented in `.Path.arc`.
        """
        if not hasattr(self, 'axes'):
            raise RuntimeError('Arcs can only be used in Axes instances')
        if not self.get_visible():
            return

        self._recompute_transform()

        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)

        # If the width and height of ellipse are not equal, take into account
        # stretching when calculating angles to draw between
        def theta_stretch(theta, scale):
            theta = np.deg2rad(theta)
            x = np.cos(theta)
            y = np.sin(theta)
            stheta = np.rad2deg(np.arctan2(scale * y, x))
            # arctan2 has the range [-pi, pi], we expect [0, 2*pi]
            return (stheta + 360) % 360

        theta1 = self.theta1
        theta2 = self.theta2

        if (
            # if we need to stretch the angles because we are distorted
            width != height
            # and we are not doing a full circle.
            #
            # 0 and 360 do not exactly round-trip through the angle
            # stretching (due to both float precision limitations and
            # the difference between the range of arctan2 [-pi, pi] and
            # this method [0, 360]) so avoid doing it if we don't have to.
            and not (theta1 != theta2 and theta1 % 360 == theta2 % 360)
        ):
            theta1 = theta_stretch(self.theta1, width / height)
            theta2 = theta_stretch(self.theta2, width / height)

        # Get width and height in pixels we need to use
        # `self.get_data_transform` rather than `self.get_transform`
        # because we want the transform from dataspace to the
        # screen space to estimate how big the arc will be in physical
        # units when rendered (the transform that we get via
        # `self.get_transform()` goes from an idealized unit-radius
        # space to screen space).
        data_to_screen_trans = self.get_data_transform()
        pwidth, pheight = (data_to_screen_trans.transform((width, height)) -
                           data_to_screen_trans.transform((0, 0)))
        inv_error = (1.0 / 1.89818e-6) * 0.5

        if pwidth < inv_error and pheight < inv_error:
            self._path = Path.arc(theta1, theta2)
            return Patch.draw(self, renderer)

        def line_circle_intersect(x0, y0, x1, y1):
            dx = x1 - x0
            dy = y1 - y0
            dr2 = dx * dx + dy * dy
            D = x0 * y1 - x1 * y0
            D2 = D * D
            discrim = dr2 - D2
            if discrim >= 0.0:
                sign_dy = np.copysign(1, dy)  # +/-1, never 0.
                sqrt_discrim = np.sqrt(discrim)
                return np.array(
                    [[(D * dy + sign_dy * dx * sqrt_discrim) / dr2,
                      (-D * dx + abs(dy) * sqrt_discrim) / dr2],
                     [(D * dy - sign_dy * dx * sqrt_discrim) / dr2,
                      (-D * dx - abs(dy) * sqrt_discrim) / dr2]])
            else:
                return np.empty((0, 2))

        def segment_circle_intersect(x0, y0, x1, y1):
            epsilon = 1e-9
            if x1 < x0:
                x0e, x1e = x1, x0
            else:
                x0e, x1e = x0, x1
            if y1 < y0:
                y0e, y1e = y1, y0
            else:
                y0e, y1e = y0, y1
            xys = line_circle_intersect(x0, y0, x1, y1)
            xs, ys = xys.T
            return xys[
                (x0e - epsilon < xs) & (xs < x1e + epsilon)
                & (y0e - epsilon < ys) & (ys < y1e + epsilon)
            ]

        # Transforms the axes box_path so that it is relative to the unit
        # circle in the same way that it is relative to the desired ellipse.
        box_path_transform = (transforms.BboxTransformTo(self.axes.bbox)
                              + self.get_transform().inverted())
        box_path = Path.unit_rectangle().transformed(box_path_transform)

        thetas = set()
        # For each of the point pairs, there is a line segment
        for p0, p1 in zip(box_path.vertices[:-1], box_path.vertices[1:]):
            xy = segment_circle_intersect(*p0, *p1)
            x, y = xy.T
            # arctan2 return [-pi, pi), the rest of our angles are in
            # [0, 360], adjust as needed.
            theta = (np.rad2deg(np.arctan2(y, x)) + 360) % 360
            thetas.update(theta[(theta1 < theta) & (theta < theta2)])
        thetas = sorted(thetas) + [theta2]
        last_theta = theta1
        theta1_rad = np.deg2rad(theta1)
        inside = box_path.contains_point(
            (np.cos(theta1_rad), np.sin(theta1_rad))
        )

        # save original path
        path_original = self._path
        for theta in thetas:
            if inside:
                self._path = Path.arc(last_theta, theta, 8)
                Patch.draw(self, renderer)
                inside = False
            else:
                inside = True
            last_theta = theta

        # restore original path
        self._path = path_original


def bbox_artist(artist, renderer, props=None, fill=True):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.

    *props* is a dict of rectangle props with the additional property
    'pad' that sets the padding around the bbox in points.
    """
    if props is None:
        props = {}
    props = props.copy()  # don't want to alter the pad externally
    pad = props.pop('pad', 4)
    pad = renderer.points_to_pixels(pad)
    bbox = artist.get_window_extent(renderer)
    r = Rectangle(
        xy=(bbox.x0 - pad / 2, bbox.y0 - pad / 2),
        width=bbox.width + pad, height=bbox.height + pad,
        fill=fill, transform=transforms.IdentityTransform(), clip_on=False)
    r.update(props)
    r.draw(renderer)


def draw_bbox(bbox, renderer, color='k', trans=None):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.
    """
    r = Rectangle(xy=(bbox.x0, bbox.y0), width=bbox.width, height=bbox.height,
                  edgecolor=color, fill=False, clip_on=False)
    if trans is not None:
        r.set_transform(trans)
    r.draw(renderer)


def _simpleprint_styles(_styles):
    """
    A helper function for the _Style class.  Given the dictionary of
    {stylename: styleclass}, return a string rep of the list of keys.
    Used to update the documentation.
    """
    return "[{}]".format("|".join(map(" '{}' ".format, sorted(_styles))))


class _Style:
    """
    A base class for the Styles. It is meant to be a container class,
    where actual styles are declared as subclass of it, and it
    provides some helper functions.
    """
    def __new__(cls, stylename, **kw):
        """Return the instance of the subclass with the given style name."""

        # The "class" should have the _style_list attribute, which is a mapping
        # of style names to style classes.

        _list = stylename.replace(" ", "").split(",")
        _name = _list[0].lower()
        try:
            _cls = cls._style_list[_name]
        except KeyError as err:
            raise ValueError("Unknown style : %s" % stylename) from err

        try:
            _args_pair = [cs.split("=") for cs in _list[1:]]
            _args = {k: float(v) for k, v in _args_pair}
        except ValueError as err:
            raise ValueError("Incorrect style argument : %s" %
                             stylename) from err
        _args.update(kw)

        return _cls(**_args)

    @classmethod
    def get_styles(cls):
        """Return a dictionary of available styles."""
        return cls._style_list

    @classmethod
    def pprint_styles(cls):
        """Return the available styles as pretty-printed string."""
        table = [('Class', 'Name', 'Attrs'),
                 *[(cls.__name__,
                    # Add backquotes, as - and | have special meaning in reST.
                    f'``{name}``',
                    # [1:-1] drops the surrounding parentheses.
                    str(inspect.signature(cls))[1:-1] or 'None')
                   for name, cls in sorted(cls._style_list.items())]]
        # Convert to rst table.
        col_len = [max(len(cell) for cell in column) for column in zip(*table)]
        table_formatstr = '  '.join('=' * cl for cl in col_len)
        rst_table = '\n'.join([
            '',
            table_formatstr,
            '  '.join(cell.ljust(cl) for cell, cl in zip(table[0], col_len)),
            table_formatstr,
            *['  '.join(cell.ljust(cl) for cell, cl in zip(row, col_len))
              for row in table[1:]],
            table_formatstr,
            '',
        ])
        return textwrap.indent(rst_table, prefix=' ' * 2)

    @classmethod
    def register(cls, name, style):
        """Register a new style."""
        if not issubclass(style, cls._Base):
            raise ValueError("%s must be a subclass of %s" % (style,
                                                              cls._Base))
        cls._style_list[name] = style


def _register_style(style_list, cls=None, *, name=None):
    """Class decorator that stashes a class in a (style) dictionary."""
    if cls is None:
        return functools.partial(_register_style, style_list, name=name)
    style_list[name or cls.__name__.lower()] = cls
    return cls


class BoxStyle(_Style):
    """
    `BoxStyle` is a container class which defines several
    boxstyle classes, which are used for `FancyBboxPatch`.

    A style object can be created as::

           BoxStyle.Round(pad=0.2)

    or::

           BoxStyle("Round", pad=0.2)

    or::

           BoxStyle("Round, pad=0.2")

    The following boxstyle classes are defined.

    %(AvailableBoxstyles)s

    An instance of any boxstyle class is an callable object,
    whose call signature is::

       __call__(self, x0, y0, width, height, mutation_size)

    and returns a `.Path` instance. *x0*, *y0*, *width* and
    *height* specify the location and size of the box to be
    drawn. *mutation_scale* determines the overall size of the
    mutation (by which I mean the transformation of the rectangle to
    the fancy box).
    """

    _style_list = {}

    @_api.deprecated("3.4")
    class _Base:
        """
        Abstract base class for styling of `.FancyBboxPatch`.

        This class is not an artist itself.  The `__call__` method returns the
        `~matplotlib.path.Path` for outlining the fancy box. The actual drawing
        is handled in `.FancyBboxPatch`.

        Subclasses may only use parameters with default values in their
        ``__init__`` method because they must be able to be initialized
        without arguments.

        Subclasses must implement the `__call__` method. It receives the
        enclosing rectangle *x0, y0, width, height* as well as the
        *mutation_size*, which scales the outline properties such as padding.
        It returns the outline of the fancy box as `.path.Path`.
        """

        @_api.deprecated("3.4")
        def transmute(self, x0, y0, width, height, mutation_size):
            """Return the `~.path.Path` outlining the given rectangle."""
            return self(self, x0, y0, width, height, mutation_size, 1)

        # This can go away once the deprecation period elapses, leaving _Base
        # as a fully abstract base class just providing docstrings, no logic.
        def __init_subclass__(cls):
            transmute = _api.deprecate_method_override(
                __class__.transmute, cls, since="3.4")
            if transmute:
                cls.__call__ = transmute
                return

            __call__ = cls.__call__

            @_api.delete_parameter("3.4", "mutation_aspect")
            def call_wrapper(
                    self, x0, y0, width, height, mutation_size,
                    mutation_aspect=_api.deprecation._deprecated_parameter):
                if mutation_aspect is _api.deprecation._deprecated_parameter:
                    # Don't trigger deprecation warning internally.
                    return __call__(self, x0, y0, width, height, mutation_size)
                else:
                    # Squeeze the given height by the aspect_ratio.
                    y0, height = y0 / mutation_aspect, height / mutation_aspect
                    path = self(x0, y0, width, height, mutation_size,
                                mutation_aspect)
                    vertices, codes = path.vertices, path.codes
                    # Restore the height.
                    vertices[:, 1] = vertices[:, 1] * mutation_aspect
                    return Path(vertices, codes)

            cls.__call__ = call_wrapper

        def __call__(self, x0, y0, width, height, mutation_size):
            """
            Given the location and size of the box, return the path of
            the box around it.

            Parameters
            ----------
            x0, y0, width, height : float
                Location and size of the box.
            mutation_size : float
                A reference scale for the mutation.

            Returns
            -------
            `~matplotlib.path.Path`
            """
            raise NotImplementedError('Derived must override')

    @_register_style(_style_list)
    class Square(_Base):
        """A square box."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            # width and height with padding added.
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height
            return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                        closed=True)

    @_register_style(_style_list)
    class Circle(_Base):
        """A circular box."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            return Path.circle((x0 + width / 2, y0 + height / 2),
                               max(width, height) / 2)

    @_register_style(_style_list)
    class LArrow(_Base):
        """A box in the shape of a left-pointing arrow."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # padding
            pad = mutation_size * self.pad
            # width and height with padding added.
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad,
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            x0 = x0 + pad / 1.4  # adjust by ~sqrt(2)

            return Path([(x0 + dxx, y0), (x1, y0), (x1, y1), (x0 + dxx, y1),
                         (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                         (x0 + dxx, y0 - dxx),  # arrow
                         (x0 + dxx, y0), (x0 + dxx, y0)],
                        closed=True)

    @_register_style(_style_list)
    class RArrow(LArrow):
        """A box in the shape of a right-pointing arrow."""

        def __call__(self, x0, y0, width, height, mutation_size):
            p = BoxStyle.LArrow.__call__(
                self, x0, y0, width, height, mutation_size)
            p.vertices[:, 0] = 2 * x0 + width - p.vertices[:, 0]
            return p

    @_register_style(_style_list)
    class DArrow(_Base):
        """A box in the shape of a two-way arrow."""
        # Modified from LArrow to add a right arrow to the bbox.

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # padding
            pad = mutation_size * self.pad
            # width and height with padding added.
            # The width is padded by the arrows, so we don't need to pad it.
            height = height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            x0 = x0 + pad / 1.4  # adjust by ~sqrt(2)

            return Path([(x0 + dxx, y0), (x1, y0),  # bot-segment
                         (x1, y0 - dxx), (x1 + dx + dxx, y0 + dx),
                         (x1, y1 + dxx),  # right-arrow
                         (x1, y1), (x0 + dxx, y1),  # top-segment
                         (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                         (x0 + dxx, y0 - dxx),  # left-arrow
                         (x0 + dxx, y0), (x0 + dxx, y0)],  # close-poly
                        closed=True)

    @_register_style(_style_list)
    class Round(_Base):
        """A box with round corners."""

        def __init__(self, pad=0.3, rounding_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*
                Radius of the corners.
            """
            self.pad = pad
            self.rounding_size = rounding_size

        def __call__(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # size of the rounding corner
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad

            width, height = width + 2 * pad, height + 2 * pad

            x0, y0 = x0 - pad, y0 - pad,
            x1, y1 = x0 + width, y0 + height

            # Round corners are implemented as quadratic Bezier, e.g.,
            # [(x0, y0-dr), (x0, y0), (x0+dr, y0)] for lower left corner.
            cp = [(x0 + dr, y0),
                  (x1 - dr, y0),
                  (x1, y0), (x1, y0 + dr),
                  (x1, y1 - dr),
                  (x1, y1), (x1 - dr, y1),
                  (x0 + dr, y1),
                  (x0, y1), (x0, y1 - dr),
                  (x0, y0 + dr),
                  (x0, y0), (x0 + dr, y0),
                  (x0 + dr, y0)]

            com = [Path.MOVETO,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.CLOSEPOLY]

            path = Path(cp, com)

            return path

    @_register_style(_style_list)
    class Round4(_Base):
        """A box with rounded edges."""

        def __init__(self, pad=0.3, rounding_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*/2
                Rounding of edges.
            """
            self.pad = pad
            self.rounding_size = rounding_size

        def __call__(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # Rounding size; defaults to half of the padding.
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad / 2.

            width = width + 2 * pad - 2 * dr
            height = height + 2 * pad - 2 * dr

            x0, y0 = x0 - pad + dr, y0 - pad + dr,
            x1, y1 = x0 + width, y0 + height

            cp = [(x0, y0),
                  (x0 + dr, y0 - dr), (x1 - dr, y0 - dr), (x1, y0),
                  (x1 + dr, y0 + dr), (x1 + dr, y1 - dr), (x1, y1),
                  (x1 - dr, y1 + dr), (x0 + dr, y1 + dr), (x0, y1),
                  (x0 - dr, y1 - dr), (x0 - dr, y0 + dr), (x0, y0),
                  (x0, y0)]

            com = [Path.MOVETO,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CLOSEPOLY]

            path = Path(cp, com)

            return path

    @_register_style(_style_list)
    class Sawtooth(_Base):
        """A box with a sawtooth outline."""

        def __init__(self, pad=0.3, tooth_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            tooth_size : float, default: *pad*/2
                Size of the sawtooth.
            """
            self.pad = pad
            self.tooth_size = tooth_size

        def _get_sawtooth_vertices(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # size of sawtooth
            if self.tooth_size is None:
                tooth_size = self.pad * .5 * mutation_size
            else:
                tooth_size = self.tooth_size * mutation_size

            tooth_size2 = tooth_size / 2
            width = width + 2 * pad - tooth_size
            height = height + 2 * pad - tooth_size

            # the sizes of the vertical and horizontal sawtooth are
            # separately adjusted to fit the given box size.
            #
            # Due to a bug in older numpy versions where round(np.float64)
            # returns a float instead of an int, convert to float before round:
            # (see gh#11810, adapted in gh#19321)
            dsx_n = round(float((width - tooth_size) / (tooth_size * 2))) * 2
            dsx = (width - tooth_size) / dsx_n
            dsy_n = round(float((height - tooth_size) / (tooth_size * 2))) * 2
            dsy = (height - tooth_size) / dsy_n

            x0, y0 = x0 - pad + tooth_size2, y0 - pad + tooth_size2
            x1, y1 = x0 + width, y0 + height

            bottom_saw_x = [
                x0,
                *(x0 + tooth_size2 + dsx * .5 * np.arange(dsx_n * 2)),
                x1 - tooth_size2,
            ]
            bottom_saw_y = [
                y0,
                *([y0 - tooth_size2, y0, y0 + tooth_size2, y0] * dsx_n),
                y0 - tooth_size2,
            ]
            right_saw_x = [
                x1,
                *([x1 + tooth_size2, x1, x1 - tooth_size2, x1] * dsx_n),
                x1 + tooth_size2,
            ]
            right_saw_y = [
                y0,
                *(y0 + tooth_size2 + dsy * .5 * np.arange(dsy_n * 2)),
                y1 - tooth_size2,
            ]
            top_saw_x = [
                x1,
                *(x1 - tooth_size2 - dsx * .5 * np.arange(dsx_n * 2)),
                x0 + tooth_size2,
            ]
            top_saw_y = [
                y1,
                *([y1 + tooth_size2, y1, y1 - tooth_size2, y1] * dsx_n),
                y1 + tooth_size2,
            ]
            left_saw_x = [
                x0,
                *([x0 - tooth_size2, x0, x0 + tooth_size2, x0] * dsy_n),
                x0 - tooth_size2,
            ]
            left_saw_y = [
                y1,
                *(y1 - tooth_size2 - dsy * .5 * np.arange(dsy_n * 2)),
                y0 + tooth_size2,
            ]

            saw_vertices = [*zip(bottom_saw_x, bottom_saw_y),
                            *zip(right_saw_x, right_saw_y),
                            *zip(top_saw_x, top_saw_y),
                            *zip(left_saw_x, left_saw_y),
                            (bottom_saw_x[0], bottom_saw_y[0])]

            return saw_vertices

        def __call__(self, x0, y0, width, height, mutation_size):
            saw_vertices = self._get_sawtooth_vertices(x0, y0, width,
                                                       height, mutation_size)
            path = Path(saw_vertices, closed=True)
            return path

    @_register_style(_style_list)
    class Roundtooth(Sawtooth):
        """A box with a rounded sawtooth outline."""

        def __call__(self, x0, y0, width, height, mutation_size):
            saw_vertices = self._get_sawtooth_vertices(x0, y0,
                                                       width, height,
                                                       mutation_size)
            # Add a trailing vertex to allow us to close the polygon correctly
            saw_vertices = np.concatenate([saw_vertices, [saw_vertices[0]]])
            codes = ([Path.MOVETO] +
                     [Path.CURVE3, Path.CURVE3] * ((len(saw_vertices)-1)//2) +
                     [Path.CLOSEPOLY])
            return Path(saw_vertices, codes)


class ConnectionStyle(_Style):
    """
    `ConnectionStyle` is a container class which defines
    several connectionstyle classes, which is used to create a path
    between two points.  These are mainly used with `FancyArrowPatch`.

    A connectionstyle object can be either created as::

           ConnectionStyle.Arc3(rad=0.2)

    or::

           ConnectionStyle("Arc3", rad=0.2)

    or::

           ConnectionStyle("Arc3, rad=0.2")

    The following classes are defined

    %(AvailableConnectorstyles)s

    An instance of any connection style class is an callable object,
    whose call signature is::

        __call__(self, posA, posB,
                 patchA=None, patchB=None,
                 shrinkA=2., shrinkB=2.)

    and it returns a `.Path` instance. *posA* and *posB* are
    tuples of (x, y) coordinates of the two points to be
    connected. *patchA* (or *patchB*) is given, the returned path is
    clipped so that it start (or end) from the boundary of the
    patch. The path is further shrunk by *shrinkA* (or *shrinkB*)
    which is given in points.
    """

    _style_list = {}

    class _Base:
        """
        A base class for connectionstyle classes. The subclass needs
        to implement a *connect* method whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The method needs to return a path connecting two
        points. This base class defines a __call__ method, and a few
        helper methods.
        """

        class SimpleEvent:
            def __init__(self, xy):
                self.x, self.y = xy

        def _clip(self, path, patchA, patchB):
            """
            Clip the path to the boundary of the patchA and patchB.
            The starting point of the path needed to be inside of the
            patchA and the end point inside the patch B. The *contains*
            methods of each patch object is utilized to test if the point
            is inside the path.
            """

            if patchA:
                def insideA(xy_display):
                    xy_event = ConnectionStyle._Base.SimpleEvent(xy_display)
                    return patchA.contains(xy_event)[0]

                try:
                    left, right = split_path_inout(path, insideA)
                except ValueError:
                    right = path

                path = right

            if patchB:
                def insideB(xy_display):
                    xy_event = ConnectionStyle._Base.SimpleEvent(xy_display)
                    return patchB.contains(xy_event)[0]

                try:
                    left, right = split_path_inout(path, insideB)
                except ValueError:
                    left = path

                path = left

            return path

        def _shrink(self, path, shrinkA, shrinkB):
            """
            Shrink the path by fixed size (in points) with shrinkA and shrinkB.
            """
            if shrinkA:
                insideA = inside_circle(*path.vertices[0], shrinkA)
                try:
                    left, path = split_path_inout(path, insideA)
                except ValueError:
                    pass
            if shrinkB:
                insideB = inside_circle(*path.vertices[-1], shrinkB)
                try:
                    path, right = split_path_inout(path, insideB)
                except ValueError:
                    pass
            return path

        def __call__(self, posA, posB,
                     shrinkA=2., shrinkB=2., patchA=None, patchB=None):
            """
            Call the *connect* method to create a path between *posA* and
            *posB*; then clip and shrink the path.
            """
            path = self.connect(posA, posB)
            clipped_path = self._clip(path, patchA, patchB)
            shrunk_path = self._shrink(clipped_path, shrinkA, shrinkB)
            return shrunk_path

    @_register_style(_style_list)
    class Arc3(_Base):
        """
        Creates a simple quadratic Bezier curve between two
        points. The curve is created so that the middle control point
        (C1) is located at the same distance from the start (C0) and
        end points(C2) and the distance of the C1 to the line
        connecting C0-C2 is *rad* times the distance of C0-C2.
        """

        def __init__(self, rad=0.):
            """
            *rad*
              curvature of the curve.
            """
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
            dx, dy = x2 - x1, y2 - y1

            f = self.rad

            cx, cy = x12 + f * dy, y12 - f * dx

            vertices = [(x1, y1),
                        (cx, cy),
                        (x2, y2)]
            codes = [Path.MOVETO,
                     Path.CURVE3,
                     Path.CURVE3]

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle3(_Base):
        """
        Creates a simple quadratic Bezier curve between two
        points. The middle control points is placed at the
        intersecting point of two lines which cross the start and
        end point, and have a slope of angleA and angleB, respectively.
        """

        def __init__(self, angleA=90, angleB=0):
            """
            *angleA*
              starting angle of the path

            *angleB*
              ending angle of the path
            """

            self.angleA = angleA
            self.angleB = angleB

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle(_Base):
        """
        Creates a piecewise continuous quadratic Bezier path between
        two points. The path has a one passing-through point placed at
        the intersecting point of two lines which cross the start
        and end point, and have a slope of angleA and angleB, respectively.
        The connecting edges are rounded with *rad*.
        """

        def __init__(self, angleA=90, angleB=0, rad=0.):
            """
            *angleA*
              starting angle of the path

            *angleB*
              ending angle of the path

            *rad*
              rounding radius of the edge
            """

            self.angleA = angleA
            self.angleB = angleB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1)]
            codes = [Path.MOVETO]

            if self.rad == 0.:
                vertices.append((cx, cy))
                codes.append(Path.LINETO)
            else:
                dx1, dy1 = x1 - cx, y1 - cy
                d1 = np.hypot(dx1, dy1)
                f1 = self.rad / d1
                dx2, dy2 = x2 - cx, y2 - cy
                d2 = np.hypot(dx2, dy2)
                f2 = self.rad / d2
                vertices.extend([(cx + dx1 * f1, cy + dy1 * f1),
                                 (cx, cy),
                                 (cx + dx2 * f2, cy + dy2 * f2)])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Arc(_Base):
        """
        Creates a piecewise continuous quadratic Bezier path between
        two points. The path can have two passing-through points, a
        point placed at the distance of armA and angle of angleA from
        point A, another point with respect to point B. The edges are
        rounded with *rad*.
        """

        def __init__(self, angleA=0, angleB=0, armA=None, armB=None, rad=0.):
            """
            *angleA* :
              starting angle of the path

            *angleB* :
              ending angle of the path

            *armA* :
              length of the starting arm

            *armB* :
              length of the ending arm

            *rad* :
              rounding radius of the edges
            """

            self.angleA = angleA
            self.angleB = angleB
            self.armA = armA
            self.armB = armB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            vertices = [(x1, y1)]
            rounded = []
            codes = [Path.MOVETO]

            if self.armA:
                cosA = math.cos(math.radians(self.angleA))
                sinA = math.sin(math.radians(self.angleA))
                # x_armA, y_armB
                d = self.armA - self.rad
                rounded.append((x1 + d * cosA, y1 + d * sinA))
                d = self.armA
                rounded.append((x1 + d * cosA, y1 + d * sinA))

            if self.armB:
                cosB = math.cos(math.radians(self.angleB))
                sinB = math.sin(math.radians(self.angleB))
                x_armB, y_armB = x2 + self.armB * cosB, y2 + self.armB * sinB

                if rounded:
                    xp, yp = rounded[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx * dx + dy * dy) ** .5

                    rounded.append((xp + self.rad * dx / dd,
                                    yp + self.rad * dy / dd))
                    vertices.extend(rounded)
                    codes.extend([Path.LINETO,
                                  Path.CURVE3,
                                  Path.CURVE3])
                else:
                    xp, yp = vertices[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx * dx + dy * dy) ** .5

                d = dd - self.rad
                rounded = [(xp + d * dx / dd, yp + d * dy / dd),
                           (x_armB, y_armB)]

            if rounded:
                xp, yp = rounded[-1]
                dx, dy = x2 - xp, y2 - yp
                dd = (dx * dx + dy * dy) ** .5

                rounded.append((xp + self.rad * dx / dd,
                                yp + self.rad * dy / dd))
                vertices.extend(rounded)
                codes.extend([Path.LINETO,
                              Path.CURVE3,
                              Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Bar(_Base):
        """
        A line with *angle* between A and B with *armA* and
        *armB*. One of the arms is extended so that they are connected in
        a right angle. The length of armA is determined by (*armA*
        + *fraction* x AB distance). Same for armB.
        """

        def __init__(self, armA=0., armB=0., fraction=0.3, angle=None):
            """
            Parameters
            ----------
            armA : float
                minimum length of armA

            armB : float
                minimum length of armB

            fraction : float
                a fraction of the distance between two points that
                will be added to armA and armB.

            angle : float or None
                angle of the connecting line (if None, parallel
                to A and B)
            """
            self.armA = armA
            self.armB = armB
            self.fraction = fraction
            self.angle = angle

        def connect(self, posA, posB):
            x1, y1 = posA
            x20, y20 = x2, y2 = posB

            theta1 = math.atan2(y2 - y1, x2 - x1)
            dx, dy = x2 - x1, y2 - y1
            dd = (dx * dx + dy * dy) ** .5
            ddx, ddy = dx / dd, dy / dd

            armA, armB = self.armA, self.armB

            if self.angle is not None:
                theta0 = np.deg2rad(self.angle)
                dtheta = theta1 - theta0
                dl = dd * math.sin(dtheta)
                dL = dd * math.cos(dtheta)
                x2, y2 = x1 + dL * math.cos(theta0), y1 + dL * math.sin(theta0)
                armB = armB - dl

                # update
                dx, dy = x2 - x1, y2 - y1
                dd2 = (dx * dx + dy * dy) ** .5
                ddx, ddy = dx / dd2, dy / dd2

            arm = max(armA, armB)
            f = self.fraction * dd + arm

            cx1, cy1 = x1 + f * ddy, y1 - f * ddx
            cx2, cy2 = x2 + f * ddy, y2 - f * ddx

            vertices = [(x1, y1),
                        (cx1, cy1),
                        (cx2, cy2),
                        (x20, y20)]
            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO]

            return Path(vertices, codes)


def _point_along_a_line(x0, y0, x1, y1, d):
    """
    Return the point on the line connecting (*x0*, *y0*) -- (*x1*, *y1*) whose
    distance from (*x0*, *y0*) is *d*.
    """
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy) ** .5
    x2, y2 = x0 - ff * dx, y0 - ff * dy

    return x2, y2


class ArrowStyle(_Style):
    """
    `ArrowStyle` is a container class which defines several
    arrowstyle classes, which is used to create an arrow path along a
    given path.  These are mainly used with `FancyArrowPatch`.

    A arrowstyle object can be either created as::

           ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy", head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")

    The following classes are defined

    %(AvailableArrowstyles)s

    An instance of any arrow style class is a callable object,
    whose call signature is::

        __call__(self, path, mutation_size, linewidth, aspect_ratio=1.)

    and it returns a tuple of a `.Path` instance and a boolean
    value. *path* is a `.Path` instance along which the arrow
    will be drawn. *mutation_size* and *aspect_ratio* have the same
    meaning as in `BoxStyle`. *linewidth* is a line width to be
    stroked. This is meant to be used to correct the location of the
    head so that it does not overshoot the destination point, but not all
    classes support it.
    """

    _style_list = {}

    class _Base:
        """
        Arrow Transmuter Base class

        ArrowTransmuterBase and its derivatives are used to make a fancy
        arrow around a given path. The __call__ method returns a path
        (which will be used to create a PathPatch instance) and a boolean
        value indicating the path is open therefore is not fillable.  This
        class is not an artist and actual drawing of the fancy arrow is
        done by the FancyArrowPatch class.

        """

        # The derived classes are required to be able to be initialized
        # w/o arguments, i.e., all its argument (except self) must have
        # the default values.

        @staticmethod
        def ensure_quadratic_bezier(path):
            """
            Some ArrowStyle class only works with a simple quadratic Bezier
            curve (created with Arc3Connection or Angle3Connector). This static
            method is to check if the provided path is a simple quadratic
            Bezier curve and returns its control points if true.
            """
            segments = list(path.iter_segments())
            if (len(segments) != 2 or segments[0][1] != Path.MOVETO or
                    segments[1][1] != Path.CURVE3):
                raise ValueError(
                    "'path' is not a valid quadratic Bezier curve")
            return [*segments[0][0], *segments[1][0]]

        def transmute(self, path, mutation_size, linewidth):
            """
            The transmute method is the very core of the ArrowStyle class and
            must be overridden in the subclasses. It receives the path object
            along which the arrow will be drawn, and the mutation_size, with
            which the arrow head etc. will be scaled. The linewidth may be
            used to adjust the path so that it does not pass beyond the given
            points. It returns a tuple of a Path instance and a boolean. The
            boolean value indicate whether the path can be filled or not. The
            return value can also be a list of paths and list of booleans of a
            same length.
            """
            raise NotImplementedError('Derived must override')

        def __call__(self, path, mutation_size, linewidth,
                     aspect_ratio=1.):
            """
            The __call__ method is a thin wrapper around the transmute method
            and takes care of the aspect ratio.
            """

            if aspect_ratio is not None:
                # Squeeze the given height by the aspect_ratio
                vertices = path.vertices / [1, aspect_ratio]
                path_shrunk = Path(vertices, path.codes)
                # call transmute method with squeezed height.
                path_mutated, fillable = self.transmute(path_shrunk,
                                                        mutation_size,
                                                        linewidth)
                if np.iterable(fillable):
                    path_list = []
                    for p in path_mutated:
                        # Restore the height
                        path_list.append(
                            Path(p.vertices * [1, aspect_ratio], p.codes))
                    return path_list, fillable
                else:
                    return path_mutated, fillable
            else:
                return self.transmute(path, mutation_size, linewidth)

    class _Curve(_Base):
        """
        A simple arrow which will work with any path instance. The
        returned path is simply concatenation of the original path + at
        most two paths representing the arrow head at the begin point and the
        at the end point. The arrow heads can be either open or closed.
        """

        def __init__(self, beginarrow=None, endarrow=None,
                     fillbegin=False, fillend=False,
                     head_length=.2, head_width=.1):
            """
            The arrows are drawn if *beginarrow* and/or *endarrow* are
            true. *head_length* and *head_width* determines the size
            of the arrow relative to the *mutation scale*.  The
            arrowhead at the begin (or end) is closed if fillbegin (or
            fillend) is True.
            """
            self.beginarrow, self.endarrow = beginarrow, endarrow
            self.head_length, self.head_width = head_length, head_width
            self.fillbegin, self.fillend = fillbegin, fillend
            super().__init__()

        def _get_arrow_wedge(self, x0, y0, x1, y1,
                             head_dist, cos_t, sin_t, linewidth):
            """
            Return the paths for arrow heads. Since arrow lines are
            drawn with capstyle=projected, The arrow goes beyond the
            desired point. This method also returns the amount of the path
            to be shrunken so that it does not overshoot.
            """

            # arrow from x0, y0 to x1, y1
            dx, dy = x0 - x1, y0 - y1

            cp_distance = np.hypot(dx, dy)

            # pad_projected : amount of pad to account the
            # overshooting of the projection of the wedge
            pad_projected = (.5 * linewidth / sin_t)

            # Account for division by zero
            if cp_distance == 0:
                cp_distance = 1

            # apply pad for projected edge
            ddx = pad_projected * dx / cp_distance
            ddy = pad_projected * dy / cp_distance

            # offset for arrow wedge
            dx = dx / cp_distance * head_dist
            dy = dy / cp_distance * head_dist

            dx1, dy1 = cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
            dx2, dy2 = cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy

            vertices_arrow = [(x1 + ddx + dx1, y1 + ddy + dy1),
                              (x1 + ddx, y1 + ddy),
                              (x1 + ddx + dx2, y1 + ddy + dy2)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO]

            return vertices_arrow, codes_arrow, ddx, ddy

        def transmute(self, path, mutation_size, linewidth):

            head_length = self.head_length * mutation_size
            head_width = self.head_width * mutation_size
            head_dist = np.hypot(head_length, head_width)
            cos_t, sin_t = head_length / head_dist, head_width / head_dist

            # begin arrow
            x0, y0 = path.vertices[0]
            x1, y1 = path.vertices[1]

            # If there is no room for an arrow and a line, then skip the arrow
            has_begin_arrow = self.beginarrow and (x0, y0) != (x1, y1)
            verticesA, codesA, ddxA, ddyA = (
                self._get_arrow_wedge(x1, y1, x0, y0,
                                      head_dist, cos_t, sin_t, linewidth)
                if has_begin_arrow
                else ([], [], 0, 0)
            )

            # end arrow
            x2, y2 = path.vertices[-2]
            x3, y3 = path.vertices[-1]

            # If there is no room for an arrow and a line, then skip the arrow
            has_end_arrow = self.endarrow and (x2, y2) != (x3, y3)
            verticesB, codesB, ddxB, ddyB = (
                self._get_arrow_wedge(x2, y2, x3, y3,
                                      head_dist, cos_t, sin_t, linewidth)
                if has_end_arrow
                else ([], [], 0, 0)
            )

            # This simple code will not work if ddx, ddy is greater than the
            # separation between vertices.
            _path = [Path(np.concatenate([[(x0 + ddxA, y0 + ddyA)],
                                          path.vertices[1:-1],
                                          [(x3 + ddxB, y3 + ddyB)]]),
                          path.codes)]
            _fillable = [False]

            if has_begin_arrow:
                if self.fillbegin:
                    p = np.concatenate([verticesA, [verticesA[0],
                                                    verticesA[0]], ])
                    c = np.concatenate([codesA, [Path.LINETO, Path.CLOSEPOLY]])
                    _path.append(Path(p, c))
                    _fillable.append(True)
                else:
                    _path.append(Path(verticesA, codesA))
                    _fillable.append(False)

            if has_end_arrow:
                if self.fillend:
                    _fillable.append(True)
                    p = np.concatenate([verticesB, [verticesB[0],
                                                    verticesB[0]], ])
                    c = np.concatenate([codesB, [Path.LINETO, Path.CLOSEPOLY]])
                    _path.append(Path(p, c))
                else:
                    _fillable.append(False)
                    _path.append(Path(verticesB, codesB))

            return _path, _fillable

    @_register_style(_style_list, name="-")
    class Curve(_Curve):
        """A simple curve without any arrow head."""

        def __init__(self):
            super().__init__(beginarrow=False, endarrow=False)

    @_register_style(_style_list, name="<-")
    class CurveA(_Curve):
        """An arrow with a head at its begin point."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=True, endarrow=False,
                             head_length=head_length, head_width=head_width)

    @_register_style(_style_list, name="->")
    class CurveB(_Curve):
        """An arrow with a head at its end point."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=False, endarrow=True,
                             head_length=head_length, head_width=head_width)

    @_register_style(_style_list, name="<->")
    class CurveAB(_Curve):
        """An arrow with heads both at the begin and the end point."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=True, endarrow=True,
                             head_length=head_length, head_width=head_width)

    @_register_style(_style_list, name="<|-")
    class CurveFilledA(_Curve):
        """An arrow with filled triangle head at the begin."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=True, endarrow=False,
                             fillbegin=True, fillend=False,
                             head_length=head_length, head_width=head_width)

    @_register_style(_style_list, name="-|>")
    class CurveFilledB(_Curve):
        """An arrow with filled triangle head at the end."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=False, endarrow=True,
                             fillbegin=False, fillend=True,
                             head_length=head_length, head_width=head_width)

    @_register_style(_style_list, name="<|-|>")
    class CurveFilledAB(_Curve):
        """An arrow with filled triangle heads at both ends."""

        def __init__(self, head_length=.4, head_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.2
                Width of the arrow head.
            """
            super().__init__(beginarrow=True, endarrow=True,
                             fillbegin=True, fillend=True,
                             head_length=head_length, head_width=head_width)

    class _Bracket(_Base):

        def __init__(self, bracketA=None, bracketB=None,
                     widthA=1., widthB=1.,
                     lengthA=0.2, lengthB=0.2,
                     angleA=None, angleB=None,
                     scaleA=None, scaleB=None):
            self.bracketA, self.bracketB = bracketA, bracketB
            self.widthA, self.widthB = widthA, widthB
            self.lengthA, self.lengthB = lengthA, lengthB
            self.angleA, self.angleB = angleA, angleB
            self.scaleA, self.scaleB = scaleA, scaleB

        def _get_bracket(self, x0, y0,
                         cos_t, sin_t, width, length):

            # arrow from x0, y0 to x1, y1
            from matplotlib.bezier import get_normal_points
            x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)

            dx, dy = length * cos_t, length * sin_t

            vertices_arrow = [(x1 + dx, y1 + dy),
                              (x1, y1),
                              (x2, y2),
                              (x2 + dx, y2 + dy)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO,
                           Path.LINETO]

            return vertices_arrow, codes_arrow

        def transmute(self, path, mutation_size, linewidth):

            if self.scaleA is None:
                scaleA = mutation_size
            else:
                scaleA = self.scaleA

            if self.scaleB is None:
                scaleB = mutation_size
            else:
                scaleB = self.scaleB

            vertices_list, codes_list = [], []

            if self.bracketA:
                x0, y0 = path.vertices[0]
                x1, y1 = path.vertices[1]
                cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
                verticesA, codesA = self._get_bracket(x0, y0, cos_t, sin_t,
                                                      self.widthA * scaleA,
                                                      self.lengthA * scaleA)
                vertices_list.append(verticesA)
                codes_list.append(codesA)

            vertices_list.append(path.vertices)
            codes_list.append(path.codes)

            if self.bracketB:
                x0, y0 = path.vertices[-1]
                x1, y1 = path.vertices[-2]
                cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
                verticesB, codesB = self._get_bracket(x0, y0, cos_t, sin_t,
                                                      self.widthB * scaleB,
                                                      self.lengthB * scaleB)
                vertices_list.append(verticesB)
                codes_list.append(codesB)

            vertices = np.concatenate(vertices_list)
            codes = np.concatenate(codes_list)

            p = Path(vertices, codes)

            return p, False

    @_register_style(_style_list, name="]-[")
    class BracketAB(_Bracket):
        """An arrow with outward square brackets at both ends."""

        def __init__(self,
                     widthA=1., lengthA=0.2, angleA=None,
                     widthB=1., lengthB=0.2, angleB=None):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.

            lengthA : float, default: 0.2
                Length of the bracket.

            angleA : float, default: None
                Angle between the bracket and the line.

            widthB : float, default: 1.0
                Width of the bracket.

            lengthB : float, default: 0.2
                Length of the bracket.

            angleB : float, default: None
                Angle between the bracket and the line.
            """
            super().__init__(True, True,
                             widthA=widthA, lengthA=lengthA, angleA=angleA,
                             widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name="]-")
    class BracketA(_Bracket):
        """An arrow with an outward square bracket at its start."""

        def __init__(self, widthA=1., lengthA=0.2, angleA=None):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.

            lengthA : float, default: 0.2
                Length of the bracket.

            angleA : float, default: None
                Angle between the bracket and the line.
            """
            super().__init__(True, None,
                             widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name="-[")
    class BracketB(_Bracket):
        """An arrow with an outward square bracket at its end."""

        def __init__(self, widthB=1., lengthB=0.2, angleB=None):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.

            lengthB : float, default: 0.2
                Length of the bracket.

            angleB : float, default: None
                Angle between the bracket and the line.
            """
            super().__init__(None, True,
                             widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name="|-|")
    class BarAB(_Bracket):
        """An arrow with vertical bars ``|`` at both ends."""

        def __init__(self,
                     widthA=1., angleA=None,
                     widthB=1., angleB=None):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.

            angleA : float, default: None
                Angle between the bracket and the line.

            widthB : float, default: 1.0
                Width of the bracket.

            angleB : float, default: None
                Angle between the bracket and the line.
            """
            super().__init__(True, True,
                             widthA=widthA, lengthA=0, angleA=angleA,
                             widthB=widthB, lengthB=0, angleB=angleB)

    @_register_style(_style_list)
    class Simple(_Base):
        """A simple arrow. Only works with a quadratic Bezier curve."""

        def __init__(self, head_length=.5, head_width=.5, tail_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.5
                Length of the arrow head.

            head_width : float, default: 0.5
                Width of the arrow head.

            tail_width : float, default: 0.2
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = \
                head_length, head_width, tail_width
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length = self.head_length * mutation_size
            in_f = inside_circle(x2, y2, head_length)
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            try:
                arrow_out, arrow_in = \
                    split_bezier_intersecting_with_closedpath(
                        arrow_path, in_f, tolerance=0.01)
            except NonIntersectingPathException:
                # if this happens, make a straight line of the head_length
                # long.
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
                arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]
                arrow_out = None

            # head
            head_width = self.head_width * mutation_size
            head_left, head_right = make_wedged_bezier2(arrow_in,
                                                        head_width / 2., wm=.5)

            # tail
            if arrow_out is not None:
                tail_width = self.tail_width * mutation_size
                tail_left, tail_right = get_parallels(arrow_out,
                                                      tail_width / 2.)

                patch_path = [(Path.MOVETO, tail_right[0]),
                              (Path.CURVE3, tail_right[1]),
                              (Path.CURVE3, tail_right[2]),
                              (Path.LINETO, head_right[0]),
                              (Path.CURVE3, head_right[1]),
                              (Path.CURVE3, head_right[2]),
                              (Path.CURVE3, head_left[1]),
                              (Path.CURVE3, head_left[0]),
                              (Path.LINETO, tail_left[2]),
                              (Path.CURVE3, tail_left[1]),
                              (Path.CURVE3, tail_left[0]),
                              (Path.LINETO, tail_right[0]),
                              (Path.CLOSEPOLY, tail_right[0]),
                              ]
            else:
                patch_path = [(Path.MOVETO, head_right[0]),
                              (Path.CURVE3, head_right[1]),
                              (Path.CURVE3, head_right[2]),
                              (Path.CURVE3, head_left[1]),
                              (Path.CURVE3, head_left[0]),
                              (Path.CLOSEPOLY, head_left[0]),
                              ]

            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    @_register_style(_style_list)
    class Fancy(_Base):
        """A fancy arrow. Only works with a quadratic Bezier curve."""

        def __init__(self, head_length=.4, head_width=.4, tail_width=.4):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.4
                Width of the arrow head.

            tail_width : float, default: 0.4
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = \
                head_length, head_width, tail_width
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length = self.head_length * mutation_size
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            # path for head
            in_f = inside_circle(x2, y2, head_length)
            try:
                path_out, path_in = split_bezier_intersecting_with_closedpath(
                    arrow_path, in_f, tolerance=0.01)
            except NonIntersectingPathException:
                # if this happens, make a straight line of the head_length
                # long.
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
                arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
                path_head = arrow_path
            else:
                path_head = path_in

            # path for head
            in_f = inside_circle(x2, y2, head_length * .8)
            path_out, path_in = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f, tolerance=0.01)
            path_tail = path_out

            # head
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(path_head,
                                                 head_width / 2.,
                                                 wm=.6)

            # tail
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(path_tail,
                                                        tail_width * .5,
                                                        w1=1., wm=0.6, w2=0.3)

            # path for head
            in_f = inside_circle(x0, y0, tail_width * .3)
            path_in, path_out = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f, tolerance=0.01)
            tail_start = path_in[-1]

            head_right, head_left = head_r, head_l
            patch_path = [(Path.MOVETO, tail_start),
                          (Path.LINETO, tail_right[0]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[2]),
                          (Path.LINETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.LINETO, tail_left[2]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[0]),
                          (Path.LINETO, tail_start),
                          (Path.CLOSEPOLY, tail_start),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    @_register_style(_style_list)
    class Wedge(_Base):
        """
        Wedge(?) shape. Only works with a quadratic Bezier curve.  The
        begin point has a width of the tail_width and the end point has a
        width of 0. At the middle, the width is shrink_factor*tail_width.
        """

        def __init__(self, tail_width=.3, shrink_factor=0.5):
            """
            Parameters
            ----------
            tail_width : float, default: 0.3
                Width of the tail.

            shrink_factor : float, default: 0.5
                Fraction of the arrow width at the middle point.
            """
            self.tail_width = tail_width
            self.shrink_factor = shrink_factor
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            b_plus, b_minus = make_wedged_bezier2(
                                    arrow_path,
                                    self.tail_width * mutation_size / 2.,
                                    wm=self.shrink_factor)

            patch_path = [(Path.MOVETO, b_plus[0]),
                          (Path.CURVE3, b_plus[1]),
                          (Path.CURVE3, b_plus[2]),
                          (Path.LINETO, b_minus[2]),
                          (Path.CURVE3, b_minus[1]),
                          (Path.CURVE3, b_minus[0]),
                          (Path.CLOSEPOLY, b_minus[0]),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True


docstring.interpd.update(
    AvailableBoxstyles=BoxStyle.pprint_styles(),
    ListBoxstyles=_simpleprint_styles(BoxStyle._style_list),
    AvailableArrowstyles=ArrowStyle.pprint_styles(),
    AvailableConnectorstyles=ConnectionStyle.pprint_styles(),
)
docstring.dedent_interpd(BoxStyle)
docstring.dedent_interpd(ArrowStyle)
docstring.dedent_interpd(ConnectionStyle)


class FancyBboxPatch(Patch):
    """
    A fancy box around a rectangle with lower left at *xy* = (*x*, *y*)
    with specified width and height.

    `.FancyBboxPatch` is similar to `.Rectangle`, but it draws a fancy box
    around the rectangle. The transformation of the rectangle box to the
    fancy box is delegated to the style classes defined in `.BoxStyle`.
    """

    _edge_default = True

    def __str__(self):
        s = self.__class__.__name__ + "((%g, %g), width=%g, height=%g)"
        return s % (self._x, self._y, self._width, self._height)

    @docstring.dedent_interpd
    @_api.delete_parameter("3.4", "bbox_transmuter", alternative="boxstyle")
    def __init__(self, xy, width, height,
                 boxstyle="round", bbox_transmuter=None,
                 mutation_scale=1, mutation_aspect=1,
                 **kwargs):
        """
        Parameters
        ----------
        xy : float, float
          The lower left corner of the box.

        width : float
            The width of the box.

        height : float
            The height of the box.

        boxstyle : str or `matplotlib.patches.BoxStyle`
            The style of the fancy box. This can either be a `.BoxStyle`
            instance or a string of the style name and optionally comma
            seprarated attributes (e.g. "Round, pad=0.2"). This string is
            passed to `.BoxStyle` to construct a `.BoxStyle` object. See
            there for a full documentation.

            The following box styles are available:

            %(AvailableBoxstyles)s

        mutation_scale : float, default: 1
            Scaling factor applied to the attributes of the box style
            (e.g. pad or rounding_size).

        mutation_aspect : float, default: 1
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it. For example, this allows different horizontal and vertical
            padding.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties

        %(Patch_kwdoc)s
        """

        super().__init__(**kwargs)

        self._x = xy[0]
        self._y = xy[1]
        self._width = width
        self._height = height

        if boxstyle == "custom":
            _api.warn_deprecated(
                "3.4", message="Support for boxstyle='custom' is deprecated "
                "since %(since)s and will be removed %(removal)s; directly "
                "pass a boxstyle instance as the boxstyle parameter instead.")
            if bbox_transmuter is None:
                raise ValueError("bbox_transmuter argument is needed with "
                                 "custom boxstyle")
            self._bbox_transmuter = bbox_transmuter
        else:
            self.set_boxstyle(boxstyle)

        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect

        self.stale = True

    @docstring.dedent_interpd
    def set_boxstyle(self, boxstyle=None, **kwargs):
        """
        Set the box style.

        Most box styles can be further configured using attributes.
        Attributes from the previous box style are not reused.

        Without argument (or with ``boxstyle=None``), the available box styles
        are returned as a human-readable string.

        Parameters
        ----------
        boxstyle : str or `matplotlib.patches.BoxStyle`
            The style of the fancy box. This can either be a `.BoxStyle`
            instance or a string of the style name and optionally comma
            seprarated attributes (e.g. "Round, pad=0.2"). This string is
            passed to `.BoxStyle` to construct a `.BoxStyle` object. See
            there for a full documentation.

            The following box styles are available:

            %(AvailableBoxstyles)s

            .. ACCEPTS: %(ListBoxstyles)s

        **kwargs
            Additional attributes for the box style. See the table above for
            supported parameters.

        Examples
        --------
        ::

            set_boxstyle("round,pad=0.2")
            set_boxstyle("round", pad=0.2)

        """
        if boxstyle is None:
            return BoxStyle.pprint_styles()

        if isinstance(boxstyle, BoxStyle._Base) or callable(boxstyle):
            self._bbox_transmuter = boxstyle
        else:
            self._bbox_transmuter = BoxStyle(boxstyle, **kwargs)
        self.stale = True

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
        """
        self._mutation_scale = scale
        self.stale = True

    def get_mutation_scale(self):
        """Return the mutation scale."""
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
        """
        self._mutation_aspect = aspect
        self.stale = True

    def get_mutation_aspect(self):
        """Return the aspect ratio of the bbox mutation."""
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # backcompat.

    def get_boxstyle(self):
        """Return the boxstyle object."""
        return self._bbox_transmuter

    def get_path(self):
        """Return the mutated path of the rectangle."""
        boxstyle = self.get_boxstyle()
        x = self._x
        y = self._y
        width = self._width
        height = self._height
        m_scale = self.get_mutation_scale()
        m_aspect = self.get_mutation_aspect()
        # Squeeze the given height by the aspect_ratio.
        y, height = y / m_aspect, height / m_aspect
        # Call boxstyle with squeezed height.
        try:
            inspect.signature(boxstyle).bind(x, y, width, height, m_scale)
        except TypeError:
            # Don't apply aspect twice.
            path = boxstyle(x, y, width, height, m_scale, 1)
            _api.warn_deprecated(
                "3.4", message="boxstyles must be callable without the "
                "'mutation_aspect' parameter since %(since)s; support for the "
                "old call signature will be removed %(removal)s.")
        else:
            path = boxstyle(x, y, width, height, m_scale)
        vertices, codes = path.vertices, path.codes
        # Restore the height.
        vertices[:, 1] = vertices[:, 1] * m_aspect
        return Path(vertices, codes)

    # Following methods are borrowed from the Rectangle class.

    def get_x(self):
        """Return the left coord of the rectangle."""
        return self._x

    def get_y(self):
        """Return the bottom coord of the rectangle."""
        return self._y

    def get_width(self):
        """Return the width of the rectangle."""
        return self._width

    def get_height(self):
        """Return the height of the rectangle."""
        return self._height

    def set_x(self, x):
        """
        Set the left coord of the rectangle.

        Parameters
        ----------
        x : float
        """
        self._x = x
        self.stale = True

    def set_y(self, y):
        """
        Set the bottom coord of the rectangle.

        Parameters
        ----------
        y : float
        """
        self._y = y
        self.stale = True

    def set_width(self, w):
        """
        Set the rectangle width.

        Parameters
        ----------
        w : float
        """
        self._width = w
        self.stale = True

    def set_height(self, h):
        """
        Set the rectangle height.

        Parameters
        ----------
        h : float
        """
        self._height = h
        self.stale = True

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle.

        Call signatures::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        Parameters
        ----------
        left, bottom : float
            The coordinates of the bottom left corner of the rectangle.
        width, height : float
            The width/height of the rectangle.
        """
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        self._x = l
        self._y = b
        self._width = w
        self._height = h
        self.stale = True

    def get_bbox(self):
        """Return the `.Bbox`."""
        return transforms.Bbox.from_bounds(self._x, self._y,
                                           self._width, self._height)


class FancyArrowPatch(Patch):
    """
    A fancy arrow patch. It draws an arrow using the `ArrowStyle`.

    The head and tail positions are fixed at the specified start and end points
    of the arrow, but the size and shape (in display coordinates) of the arrow
    does not change when the axis is moved or zoomed.
    """
    _edge_default = True

    def __str__(self):
        if self._posA_posB is not None:
            (x1, y1), (x2, y2) = self._posA_posB
            return f"{type(self).__name__}(({x1:g}, {y1:g})->({x2:g}, {y2:g}))"
        else:
            return f"{type(self).__name__}({self._path_original})"

    @docstring.dedent_interpd
    @_api.delete_parameter("3.4", "dpi_cor")
    def __init__(self, posA=None, posB=None, path=None,
                 arrowstyle="simple", connectionstyle="arc3",
                 patchA=None, patchB=None,
                 shrinkA=2, shrinkB=2,
                 mutation_scale=1, mutation_aspect=1,
                 dpi_cor=1,
                 **kwargs):
        """
        There are two ways for defining an arrow:

        - If *posA* and *posB* are given, a path connecting two points is
          created according to *connectionstyle*. The path will be
          clipped with *patchA* and *patchB* and further shrunken by
          *shrinkA* and *shrinkB*. An arrow is drawn along this
          resulting path using the *arrowstyle* parameter.

        - Alternatively if *path* is provided, an arrow is drawn along this
          path and *patchA*, *patchB*, *shrinkA*, and *shrinkB* are ignored.

        Parameters
        ----------
        posA, posB : (float, float), default: None
            (x, y) coordinates of arrow tail and arrow head respectively.

        path : `~matplotlib.path.Path`, default: None
            If provided, an arrow is drawn along this path and *patchA*,
            *patchB*, *shrinkA*, and *shrinkB* are ignored.

        arrowstyle : str or `.ArrowStyle`, default: 'simple'
            The `.ArrowStyle` with which the fancy arrow is drawn.  If a
            string, it should be one of the available arrowstyle names, with
            optional comma-separated attributes.  The optional attributes are
            meant to be scaled with the *mutation_scale*.  The following arrow
            styles are available:

            %(AvailableArrowstyles)s

        connectionstyle : str or `.ConnectionStyle` or None, optional, \
default: 'arc3'
            The `.ConnectionStyle` with which *posA* and *posB* are connected.
            If a string, it should be one of the available connectionstyle
            names, with optional comma-separated attributes.  The following
            connection styles are available:

            %(AvailableConnectorstyles)s

        patchA, patchB : `.Patch`, default: None
            Head and tail patches, respectively.

        shrinkA, shrinkB : float, default: 2
            Shrinking factor of the tail and head of the arrow respectively.

        mutation_scale : float, default: 1
            Value with which attributes of *arrowstyle* (e.g., *head_length*)
            will be scaled.

        mutation_aspect : None or float, default: None
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it.

        dpi_cor : float, default: 1
            dpi_cor is currently used for linewidth-related things and shrink
            factor. Mutation scale is affected by this.  Deprecated.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties, optional
            Here is a list of available `.Patch` properties:

        %(Patch_kwdoc)s

            In contrast to other patches, the default ``capstyle`` and
            ``joinstyle`` for `FancyArrowPatch` are set to ``"round"``.
        """
        # Traditionally, the cap- and joinstyle for FancyArrowPatch are round
        kwargs.setdefault("joinstyle", "round")
        kwargs.setdefault("capstyle", "round")

        super().__init__(**kwargs)

        if posA is not None and posB is not None and path is None:
            self._posA_posB = [posA, posB]

            if connectionstyle is None:
                connectionstyle = "arc3"
            self.set_connectionstyle(connectionstyle)

        elif posA is None and posB is None and path is not None:
            self._posA_posB = None
        else:
            raise ValueError("Either posA and posB, or path need to provided")

        self.patchA = patchA
        self.patchB = patchB
        self.shrinkA = shrinkA
        self.shrinkB = shrinkB

        self._path_original = path

        self.set_arrowstyle(arrowstyle)

        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect

        self._dpi_cor = dpi_cor

    @_api.deprecated("3.4")
    def set_dpi_cor(self, dpi_cor):
        """
        dpi_cor is currently used for linewidth-related things and
        shrink factor. Mutation scale is affected by this.

        Parameters
        ----------
        dpi_cor : float
        """
        self._dpi_cor = dpi_cor
        self.stale = True

    @_api.deprecated("3.4")
    def get_dpi_cor(self):
        """
        dpi_cor is currently used for linewidth-related things and
        shrink factor. Mutation scale is affected by this.

        Returns
        -------
        scalar
        """
        return self._dpi_cor

    def set_positions(self, posA, posB):
        """
        Set the begin and end positions of the connecting path.

        Parameters
        ----------
        posA, posB : None, tuple
            (x, y) coordinates of arrow tail and arrow head respectively. If
            `None` use current value.
        """
        if posA is not None:
            self._posA_posB[0] = posA
        if posB is not None:
            self._posA_posB[1] = posB
        self.stale = True

    def set_patchA(self, patchA):
        """
        Set the tail patch.

        Parameters
        ----------
        patchA : `.patches.Patch`
        """
        self.patchA = patchA
        self.stale = True

    def set_patchB(self, patchB):
        """
        Set the head patch.

        Parameters
        ----------
        patchB : `.patches.Patch`
        """
        self.patchB = patchB
        self.stale = True

    def set_connectionstyle(self, connectionstyle, **kw):
        """
        Set the connection style. Old attributes are forgotten.

        Parameters
        ----------
        connectionstyle : str or `.ConnectionStyle` or None, optional
            Can be a string with connectionstyle name with
            optional comma-separated attributes, e.g.::

                set_connectionstyle("arc,angleA=0,armA=30,rad=10")

            Alternatively, the attributes can be provided as keywords, e.g.::

                set_connectionstyle("arc", angleA=0,armA=30,rad=10)

            Without any arguments (or with ``connectionstyle=None``), return
            available styles as a list of strings.
        """

        if connectionstyle is None:
            return ConnectionStyle.pprint_styles()

        if (isinstance(connectionstyle, ConnectionStyle._Base) or
                callable(connectionstyle)):
            self._connector = connectionstyle
        else:
            self._connector = ConnectionStyle(connectionstyle, **kw)
        self.stale = True

    def get_connectionstyle(self):
        """Return the `ConnectionStyle` used."""
        return self._connector

    def set_arrowstyle(self, arrowstyle=None, **kw):
        """
        Set the arrow style. Old attributes are forgotten. Without arguments
        (or with ``arrowstyle=None``) returns available box styles as a list of
        strings.

        Parameters
        ----------
        arrowstyle : None or ArrowStyle or str, default: None
            Can be a string with arrowstyle name with optional comma-separated
            attributes, e.g.::

                set_arrowstyle("Fancy,head_length=0.2")

            Alternatively attributes can be provided as keywords, e.g.::

                set_arrowstyle("fancy", head_length=0.2)

        """

        if arrowstyle is None:
            return ArrowStyle.pprint_styles()

        if isinstance(arrowstyle, ArrowStyle._Base):
            self._arrow_transmuter = arrowstyle
        else:
            self._arrow_transmuter = ArrowStyle(arrowstyle, **kw)
        self.stale = True

    def get_arrowstyle(self):
        """Return the arrowstyle object."""
        return self._arrow_transmuter

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
        """
        self._mutation_scale = scale
        self.stale = True

    def get_mutation_scale(self):
        """
        Return the mutation scale.

        Returns
        -------
        scalar
        """
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
        """
        self._mutation_aspect = aspect
        self.stale = True

    def get_mutation_aspect(self):
        """Return the aspect ratio of the bbox mutation."""
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # backcompat.

    def get_path(self):
        """
        Return the path of the arrow in the data coordinates. Use
        get_path_in_displaycoord() method to retrieve the arrow path
        in display coordinates.
        """
        _path, fillable = self.get_path_in_displaycoord()
        if np.iterable(fillable):
            _path = Path.make_compound_path(*_path)
        return self.get_transform().inverted().transform_path(_path)

    def get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor

        if self._posA_posB is not None:
            posA = self._convert_xy_units(self._posA_posB[0])
            posB = self._convert_xy_units(self._posA_posB[1])
            (posA, posB) = self.get_transform().transform((posA, posB))
            _path = self.get_connectionstyle()(posA, posB,
                                               patchA=self.patchA,
                                               patchB=self.patchB,
                                               shrinkA=self.shrinkA * dpi_cor,
                                               shrinkB=self.shrinkB * dpi_cor
                                               )
        else:
            _path = self.get_transform().transform_path(self._path_original)

        _path, fillable = self.get_arrowstyle()(
            _path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect())

        # if not fillable:
        #    self._fill = False

        return _path, fillable

    def draw(self, renderer):
        if not self.get_visible():
            return

        with self._bind_draw_path_function(renderer) as draw_path:

            # FIXME : dpi_cor is for the dpi-dependency of the linewidth. There
            # could be room for improvement.  Maybe get_path_in_displaycoord
            # could take a renderer argument, but get_path should be adapted
            # too.
            self._dpi_cor = renderer.points_to_pixels(1.)
            path, fillable = self.get_path_in_displaycoord()

            if not np.iterable(fillable):
                path = [path]
                fillable = [fillable]

            affine = transforms.IdentityTransform()

            for p, f in zip(path, fillable):
                draw_path(
                    p, affine,
                    self._facecolor if f and self._facecolor[3] else None)


class ConnectionPatch(FancyArrowPatch):
    """A patch that connects two points (possibly in different axes)."""

    def __str__(self):
        return "ConnectionPatch((%g, %g), (%g, %g))" % \
               (self.xy1[0], self.xy1[1], self.xy2[0], self.xy2[1])

    @docstring.dedent_interpd
    @_api.delete_parameter("3.4", "dpi_cor")
    def __init__(self, xyA, xyB, coordsA, coordsB=None,
                 axesA=None, axesB=None,
                 arrowstyle="-",
                 connectionstyle="arc3",
                 patchA=None,
                 patchB=None,
                 shrinkA=0.,
                 shrinkB=0.,
                 mutation_scale=10.,
                 mutation_aspect=None,
                 clip_on=False,
                 dpi_cor=1.,
                 **kwargs):
        """
        Connect point *xyA* in *coordsA* with point *xyB* in *coordsB*.

        Valid keys are

        ===============  ======================================================
        Key              Description
        ===============  ======================================================
        arrowstyle       the arrow style
        connectionstyle  the connection style
        relpos           default is (0.5, 0.5)
        patchA           default is bounding box of the text
        patchB           default is None
        shrinkA          default is 2 points
        shrinkB          default is 2 points
        mutation_scale   default is text size (in points)
        mutation_aspect  default is 1.
        ?                any key for `matplotlib.patches.PathPatch`
        ===============  ======================================================

        *coordsA* and *coordsB* are strings that indicate the
        coordinates of *xyA* and *xyB*.

        ==================== ==================================================
        Property             Description
        ==================== ==================================================
        'figure points'      points from the lower left corner of the figure
        'figure pixels'      pixels from the lower left corner of the figure
        'figure fraction'    0, 0 is lower left of figure and 1, 1 is upper
                             right
        'subfigure points'   points from the lower left corner of the subfigure
        'subfigure pixels'   pixels from the lower left corner of the subfigure
        'subfigure fraction' fraction of the subfigure, 0, 0 is lower left.
        'axes points'        points from lower left corner of axes
        'axes pixels'        pixels from lower left corner of axes
        'axes fraction'      0, 0 is lower left of axes and 1, 1 is upper right
        'data'               use the coordinate system of the object being
                             annotated (default)
        'offset points'      offset (in points) from the *xy* value
        'polar'              you can specify *theta*, *r* for the annotation,
                             even in cartesian plots.  Note that if you are
                             using a polar axes, you do not need to specify
                             polar for the coordinate system since that is the
                             native "data" coordinate system.
        ==================== ==================================================

        Alternatively they can be set to any valid
        `~matplotlib.transforms.Transform`.

        Note that 'subfigure pixels' and 'figure pixels' are the same
        for the parent figure, so users who want code that is usable in
        a subfigure can use 'subfigure pixels'.

        .. note::

           Using `ConnectionPatch` across two `~.axes.Axes` instances
           is not directly compatible with :doc:`constrained layout
           </tutorials/intermediate/constrainedlayout_guide>`. Add the artist
           directly to the `.Figure` instead of adding it to a specific Axes,
           or exclude it from the layout using ``con.set_in_layout(False)``.

           .. code-block:: default

              fig, ax = plt.subplots(1, 2, constrained_layout=True)
              con = ConnectionPatch(..., axesA=ax[0], axesB=ax[1])
              fig.add_artist(con)

        """
        if coordsB is None:
            coordsB = coordsA
        # we'll draw ourself after the artist we annotate by default
        self.xy1 = xyA
        self.xy2 = xyB
        self.coords1 = coordsA
        self.coords2 = coordsB

        self.axesA = axesA
        self.axesB = axesB

        super().__init__(posA=(0, 0), posB=(1, 1),
                         arrowstyle=arrowstyle,
                         connectionstyle=connectionstyle,
                         patchA=patchA, patchB=patchB,
                         shrinkA=shrinkA, shrinkB=shrinkB,
                         mutation_scale=mutation_scale,
                         mutation_aspect=mutation_aspect,
                         clip_on=clip_on,
                         **kwargs)
        self._dpi_cor = dpi_cor

        # if True, draw annotation only if self.xy is inside the axes
        self._annotation_clip = None

    def _get_xy(self, xy, s, axes=None):
        """Calculate the pixel position of given point."""
        s0 = s  # For the error message, if needed.
        if axes is None:
            axes = self.axes
        xy = np.array(xy)
        if s in ["figure points", "axes points"]:
            xy *= self.figure.dpi / 72
            s = s.replace("points", "pixels")
        elif s == "figure fraction":
            s = self.figure.transFigure
        elif s == "subfigure fraction":
            s = self.figure.transSubfigure
        elif s == "axes fraction":
            s = axes.transAxes
        x, y = xy

        if s == 'data':
            trans = axes.transData
            x = float(self.convert_xunits(x))
            y = float(self.convert_yunits(y))
            return trans.transform((x, y))
        elif s == 'offset points':
            if self.xycoords == 'offset points':  # prevent recursion
                return self._get_xy(self.xy, 'data')
            return (
                self._get_xy(self.xy, self.xycoords)  # converted data point
                + xy * self.figure.dpi / 72)  # converted offset
        elif s == 'polar':
            theta, r = x, y
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            trans = axes.transData
            return trans.transform((x, y))
        elif s == 'figure pixels':
            # pixels from the lower left corner of the figure
            bb = self.figure.figbbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'subfigure pixels':
            # pixels from the lower left corner of the figure
            bb = self.figure.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'axes pixels':
            # pixels from the lower left corner of the axes
            bb = axes.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif isinstance(s, transforms.Transform):
            return s.transform(xy)
        else:
            raise ValueError(f"{s0} is not a valid coordinate transformation")

    def set_annotation_clip(self, b):
        """
        Set the clipping behavior.

        Parameters
        ----------
        b : bool or None

            - *False*: The annotation will always be drawn regardless of its
              position.
            - *True*: The annotation will only be drawn if ``self.xy`` is
              inside the axes.
            - *None*: The annotation will only be drawn if ``self.xy`` is
              inside the axes and  ``self.xycoords == "data"``.
        """
        self._annotation_clip = b
        self.stale = True

    def get_annotation_clip(self):
        """
        Return the clipping behavior.

        See `.set_annotation_clip` for the meaning of the return value.
        """
        return self._annotation_clip

    def get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor
        posA = self._get_xy(self.xy1, self.coords1, self.axesA)
        posB = self._get_xy(self.xy2, self.coords2, self.axesB)
        path = self.get_connectionstyle()(
            posA, posB,
            patchA=self.patchA, patchB=self.patchB,
            shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor,
        )
        path, fillable = self.get_arrowstyle()(
            path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect()
        )
        return path, fillable

    def _check_xy(self, renderer):
        """Check whether the annotation needs to be drawn."""

        b = self.get_annotation_clip()

        if b or (b is None and self.coords1 == "data"):
            xy_pixel = self._get_xy(self.xy1, self.coords1, self.axesA)
            if self.axesA is None:
                axes = self.axes
            else:
                axes = self.axesA
            if not axes.contains_point(xy_pixel):
                return False

        if b or (b is None and self.coords2 == "data"):
            xy_pixel = self._get_xy(self.xy2, self.coords2, self.axesB)
            if self.axesB is None:
                axes = self.axes
            else:
                axes = self.axesB
            if not axes.contains_point(xy_pixel):
                return False

        return True

    def draw(self, renderer):
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        super().draw(renderer)
