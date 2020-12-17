r"""
A module for dealing with the polylines used throughout Matplotlib.

The primary class for polyline handling in Matplotlib is `Path`.  Almost all
vector drawing makes use of `Path`\s somewhere in the drawing pipeline.

Whilst a `Path` instance itself cannot be drawn, some `.Artist` subclasses,
such as `.PathPatch` and `.PathCollection`, can be used for convenient `Path`
visualisation.
"""

from collections import namedtuple
from functools import lru_cache
from weakref import WeakValueDictionary

import numpy as np

import matplotlib as mpl
from . import _api, _path, cbook
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment


VertexInfo = namedtuple('VertexInfo', 'apex incidence_angle corner_angle')
VertexInfo.__doc__ = r"""
Holds information necessary to ascertain the bounding box of a vertex once it's
been stroked at a given ``markeredgewidth``.

Attributes
----------
apex : Tuple[float,float]
    The position of the vertex.
incidence_angle : float, in [-pi, pi]
    For vertices with one incoming line, the orientation of the tangent vector
    to the line at its tip (as returned by `np.arctan2`). For corners (vertices
    with two incoming lines) the orientation of the corner's bisector should be
    used instead. All orientation angles should be computed assuming the line
    or bisector "points" towards the corner's apex.
corner_angle : float, in [0, pi]
    The internal angle of the corner, where np.pi is a straight line, and 0
    is retracing exactly the way you came. None can be used to signify that
    the line ends there (i.e. no corner).
"""


class Path:
    """
    A series of possibly disconnected, possibly closed, line and curve
    segments.

    The underlying storage is made up of two parallel numpy arrays:

    - *vertices*: an Nx2 float array of vertices
    - *codes*: an N-length uint8 array of vertex types, or None

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices as well as three codes ``CURVE3``.

    The code types are:

    - ``STOP``   :  1 vertex (ignored)
        A marker for the end of the entire path (currently not required and
        ignored)

    - ``MOVETO`` :  1 vertex
        Pick up the pen and move to the given vertex.

    - ``LINETO`` :  1 vertex
        Draw a line from the current position to the given vertex.

    - ``CURVE3`` :  1 control point, 1 endpoint
        Draw a quadratic Bezier curve from the current position, with the given
        control point, to the given end point.

    - ``CURVE4`` :  2 control points, 1 endpoint
        Draw a cubic Bezier curve from the current position, with the given
        control points, to the given end point.

    - ``CLOSEPOLY`` : 1 vertex (ignored)
        Draw a line segment to the start point of the current polyline.

    If *codes* is None, it is interpreted as a ``MOVETO`` followed by a series
    of ``LINETO``.

    Users of Path objects should not access the vertices and codes arrays
    directly.  Instead, they should use `iter_segments` or `cleaned` to get the
    vertex/code pairs.  This helps, in particular, to consistently handle the
    case of *codes* being None.

    Some behavior of Path objects can be controlled by rcParams. See the
    rcParams whose keys start with 'path.'.

    .. note::

        The vertices and codes arrays should be treated as
        immutable -- there are a number of optimizations and assumptions
        made up front in the constructor that will not change when the
        data changes.
    """

    code_type = np.uint8

    # Path codes
    STOP = code_type(0)         # 1 vertex
    MOVETO = code_type(1)       # 1 vertex
    LINETO = code_type(2)       # 1 vertex
    CURVE3 = code_type(3)       # 2 vertices
    CURVE4 = code_type(4)       # 3 vertices
    CLOSEPOLY = code_type(79)   # 1 vertex

    #: A dictionary mapping Path codes to the number of vertices that the
    #: code expects.
    NUM_VERTICES_FOR_CODE = {STOP: 1,
                             MOVETO: 1,
                             LINETO: 1,
                             CURVE3: 2,
                             CURVE4: 3,
                             CLOSEPOLY: 1}

    def __init__(self, vertices, codes=None, _interpolation_steps=1,
                 closed=False, readonly=False):
        """
        Create a new path with the given vertices and codes.

        Parameters
        ----------
        vertices : array-like
            The ``(N, 2)`` float array, masked array or sequence of pairs
            representing the vertices of the path.

            If *vertices* contains masked values, they will be converted
            to NaNs which are then handled correctly by the Agg
            PathIterator and other consumers of path data, such as
            :meth:`iter_segments`.
        codes : array-like or None, optional
            n-length array integers representing the codes of the path.
            If not None, codes must be the same length as vertices.
            If None, *vertices* will be treated as a series of line segments.
        _interpolation_steps : int, optional
            Used as a hint to certain projections, such as Polar, that this
            path should be linearly interpolated immediately before drawing.
            This attribute is primarily an implementation detail and is not
            intended for public use.
        closed : bool, optional
            If *codes* is None and closed is True, vertices will be treated as
            line segments of a closed polygon.  Note that the last vertex will
            then be ignored (as the corresponding code will be set to
            CLOSEPOLY).
        readonly : bool, optional
            Makes the path behave in an immutable way and sets the vertices
            and codes as read-only arrays.
        """
        vertices = _to_unmasked_float_array(vertices)
        _api.check_shape((None, 2), vertices=vertices)

        if codes is not None:
            codes = np.asarray(codes, self.code_type)
            if codes.ndim != 1 or len(codes) != len(vertices):
                raise ValueError("'codes' must be a 1D list or array with the "
                                 "same length of 'vertices'. "
                                 f"Your vertices have shape {vertices.shape} "
                                 f"but your codes have shape {codes.shape}")
            if len(codes) and codes[0] != self.MOVETO:
                raise ValueError("The first element of 'code' must be equal "
                                 f"to 'MOVETO' ({self.MOVETO}).  "
                                 f"Your first code is {codes[0]}")
        elif closed and len(vertices):
            codes = np.empty(len(vertices), dtype=self.code_type)
            codes[0] = self.MOVETO
            codes[1:-1] = self.LINETO
            codes[-1] = self.CLOSEPOLY

        self._vertices = vertices
        self._codes = codes
        self._interpolation_steps = _interpolation_steps
        self._update_values()

        if readonly:
            self._vertices.flags.writeable = False
            if self._codes is not None:
                self._codes.flags.writeable = False
            self._readonly = True
        else:
            self._readonly = False

    @classmethod
    def _fast_from_codes_and_verts(cls, verts, codes, internals_from=None):
        """
        Creates a Path instance without the expense of calling the constructor.

        Parameters
        ----------
        verts : numpy array
        codes : numpy array
        internals_from : Path or None
            If not None, another `Path` from which the attributes
            ``should_simplify``, ``simplify_threshold``, and
            ``interpolation_steps`` will be copied.  Note that ``readonly`` is
            never copied, and always set to ``False`` by this constructor.
        """
        pth = cls.__new__(cls)
        pth._vertices = _to_unmasked_float_array(verts)
        pth._codes = codes
        pth._readonly = False
        if internals_from is not None:
            pth._should_simplify = internals_from._should_simplify
            pth._simplify_threshold = internals_from._simplify_threshold
            pth._interpolation_steps = internals_from._interpolation_steps
        else:
            pth._should_simplify = True
            pth._simplify_threshold = mpl.rcParams['path.simplify_threshold']
            pth._interpolation_steps = 1
        return pth

    def _update_values(self):
        self._simplify_threshold = mpl.rcParams['path.simplify_threshold']
        self._should_simplify = (
            self._simplify_threshold > 0 and
            mpl.rcParams['path.simplify'] and
            len(self._vertices) >= 128 and
            (self._codes is None or np.all(self._codes <= Path.LINETO))
        )

    @property
    def vertices(self):
        """
        The list of vertices in the `Path` as an Nx2 numpy array.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        if self._readonly:
            raise AttributeError("Can't set vertices on a readonly Path")
        self._vertices = vertices
        self._update_values()

    @property
    def codes(self):
        """
        The list of codes in the `Path` as a 1D numpy array.  Each
        code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4`
        or `CLOSEPOLY`.  For codes that correspond to more than one
        vertex (`CURVE3` and `CURVE4`), that code will be repeated so
        that the length of `self.vertices` and `self.codes` is always
        the same.
        """
        return self._codes

    @codes.setter
    def codes(self, codes):
        if self._readonly:
            raise AttributeError("Can't set codes on a readonly Path")
        self._codes = codes
        self._update_values()

    @property
    def simplify_threshold(self):
        """
        The fraction of a pixel difference below which vertices will
        be simplified out.
        """
        return self._simplify_threshold

    @simplify_threshold.setter
    def simplify_threshold(self, threshold):
        self._simplify_threshold = threshold

    @property
    def should_simplify(self):
        """
        `True` if the vertices array should be simplified.
        """
        return self._should_simplify

    @should_simplify.setter
    def should_simplify(self, should_simplify):
        self._should_simplify = should_simplify

    @property
    def readonly(self):
        """
        `True` if the `Path` is read-only.
        """
        return self._readonly

    def __copy__(self):
        """
        Return a shallow copy of the `Path`, which will share the
        vertices and codes with the source `Path`.
        """
        import copy
        return copy.copy(self)

    copy = __copy__

    def __deepcopy__(self, memo=None):
        """
        Return a deepcopy of the `Path`.  The `Path` will not be
        readonly, even if the source `Path` is.
        """
        try:
            codes = self.codes.copy()
        except AttributeError:
            codes = None
        return self.__class__(
            self.vertices.copy(), codes,
            _interpolation_steps=self._interpolation_steps)

    deepcopy = __deepcopy__

    @classmethod
    def make_compound_path_from_polys(cls, XY):
        """
        Make a compound path object to draw a number
        of polygons with equal numbers of sides XY is a (numpolys x
        numsides x 2) numpy array of vertices.  Return object is a
        :class:`Path`

        .. plot:: gallery/misc/histogram_path.py

        """

        # for each poly: 1 for the MOVETO, (numsides-1) for the LINETO, 1 for
        # the CLOSEPOLY; the vert for the closepoly is ignored but we still
        # need it to keep the codes aligned with the vertices
        numpolys, numsides, two = XY.shape
        if two != 2:
            raise ValueError("The third dimension of 'XY' must be 2")
        stride = numsides + 1
        nverts = numpolys * stride
        verts = np.zeros((nverts, 2))
        codes = np.full(nverts, cls.LINETO, dtype=cls.code_type)
        codes[0::stride] = cls.MOVETO
        codes[numsides::stride] = cls.CLOSEPOLY
        for i in range(numsides):
            verts[i::stride] = XY[:, i]

        return cls(verts, codes)

    @classmethod
    def make_compound_path(cls, *args):
        """
        Make a compound path from a list of Path objects. Blindly removes all
        Path.STOP control points.
        """
        # Handle an empty list in args (i.e. no args).
        if not args:
            return Path(np.empty([0, 2], dtype=np.float32))
        vertices = np.concatenate([x.vertices for x in args])
        codes = np.empty(len(vertices), dtype=cls.code_type)
        i = 0
        for path in args:
            if path.codes is None:
                codes[i] = cls.MOVETO
                codes[i + 1:i + len(path.vertices)] = cls.LINETO
            else:
                codes[i:i + len(path.codes)] = path.codes
            i += len(path.vertices)
        # remove STOP's, since internal STOPs are a bug
        not_stop_mask = codes != cls.STOP
        vertices = vertices[not_stop_mask, :]
        codes = codes[not_stop_mask]

        return cls(vertices, codes)

    def __repr__(self):
        return "Path(%r, %r)" % (self.vertices, self.codes)

    def __len__(self):
        return len(self.vertices)

    def iter_segments(self, transform=None, remove_nans=True, clip=None,
                      snap=False, stroke_width=1.0, simplify=None,
                      curves=True, sketch=None):
        """
        Iterate over all curve segments in the path.

        Each iteration returns a pair ``(vertices, code)``, where ``vertices``
        is a sequence of 1-3 coordinate pairs, and ``code`` is a `Path` code.

        Additionally, this method can provide a number of standard cleanups and
        conversions to the path.

        Parameters
        ----------
        transform : None or :class:`~matplotlib.transforms.Transform`
            If not None, the given affine transformation will be applied to the
            path.
        remove_nans : bool, optional
            Whether to remove all NaNs from the path and skip over them using
            MOVETO commands.
        clip : None or (float, float, float, float), optional
            If not None, must be a four-tuple (x1, y1, x2, y2)
            defining a rectangle in which to clip the path.
        snap : None or bool, optional
            If True, snap all nodes to pixels; if False, don't snap them.
            If None, snap if the path contains only segments
            parallel to the x or y axes, and no more than 1024 of them.
        stroke_width : float, optional
            The width of the stroke being drawn (used for path snapping).
        simplify : None or bool, optional
            Whether to simplify the path by removing vertices
            that do not affect its appearance.  If None, use the
            :attr:`should_simplify` attribute.  See also :rc:`path.simplify`
            and :rc:`path.simplify_threshold`.
        curves : bool, optional
            If True, curve segments will be returned as curve segments.
            If False, all curves will be converted to line segments.
        sketch : None or sequence, optional
            If not None, must be a 3-tuple of the form
            (scale, length, randomness), representing the sketch parameters.
        """
        if not len(self):
            return

        cleaned = self.cleaned(transform=transform,
                               remove_nans=remove_nans, clip=clip,
                               snap=snap, stroke_width=stroke_width,
                               simplify=simplify, curves=curves,
                               sketch=sketch)

        # Cache these object lookups for performance in the loop.
        NUM_VERTICES_FOR_CODE = self.NUM_VERTICES_FOR_CODE
        STOP = self.STOP

        vertices = iter(cleaned.vertices)
        codes = iter(cleaned.codes)
        for curr_vertices, code in zip(vertices, codes):
            if code == STOP:
                break
            extra_vertices = NUM_VERTICES_FOR_CODE[code] - 1
            if extra_vertices:
                for i in range(extra_vertices):
                    next(codes)
                    curr_vertices = np.append(curr_vertices, next(vertices))
            yield curr_vertices, code

    def iter_bezier(self, **kwargs):
        """
        Iterate over each bezier curve (lines included) in a Path.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`.

        Yields
        ------
        B : matplotlib.bezier.BezierSegment
            The bezier curves that make up the current path. Note in particular
            that freestanding points are bezier curves of order 0, and lines
            are bezier curves of order 1 (with two control points).
        code : Path.code_type
            The code describing what kind of curve is being returned.
            Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE4 correspond to
            bezier curves with 1, 2, 3, and 4 control points (respectively).
            Path.CLOSEPOLY is a Path.LINETO with the control points correctly
            chosen based on the start/end points of the current stroke.
        """
        first_vert = None
        prev_vert = None
        for verts, code in self.iter_segments(**kwargs):
            if first_vert is None:
                if code != Path.MOVETO:
                    raise ValueError("Malformed path, must start with MOVETO.")
            if code == Path.MOVETO:  # a point is like "CURVE1"
                first_vert = verts
                yield BezierSegment(np.array([first_vert])), code
            elif code == Path.LINETO:  # "CURVE2"
                yield BezierSegment(np.array([prev_vert, verts])), code
            elif code == Path.CURVE3:
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:]])), code
            elif code == Path.CURVE4:
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:4], verts[4:]])), code
            elif code == Path.CLOSEPOLY:
                yield BezierSegment(np.array([prev_vert, first_vert])), code
            elif code == Path.STOP:
                return
            else:
                raise ValueError("Invalid Path.code_type: " + str(code))
            prev_vert = verts[-2:]

    def iter_angles(self, **kwargs):
        """
        Iterate over `.VertexInfo` for each vertex in the path.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`

        Yields
        ------
        vinfo : `.VertexInfo`
            Measure of the vertex's position, orientation, and angle (if it's
            the apex of a corner). Useful in order to determine how the corner
            affects the bbox of the curve.
        """
        first_tan_angle = None
        first_vertex = None
        prev_tan_angle = None
        prev_vertex = None
        is_capped = False
        for B, code in self.iter_bezier(**kwargs):
            if code == Path.MOVETO:
                # deal with capping ends of previous polyline, if it exists
                if prev_tan_angle is not None and is_capped:
                    cap_angles = [first_tan_angle, prev_tan_angle]
                    cap_vertices = [first_vertex, prev_vertex]
                    for cap_angle, cap_vertex in zip(cap_angles, cap_vertices):
                        yield VertexInfo(cap_vertex, cap_angle, None)
                first_tan_angle = None
                prev_tan_angle = None
                first_vertex = B.control_points[0]
                prev_vertex = first_vertex
                # lines end in a cap by default unless a CLOSEPOLY is observed
                is_capped = True
                continue
            if code == Path.CLOSEPOLY:
                is_capped = False
                if prev_tan_angle is None:
                    raise ValueError("Misformed path, cannot close poly with "
                                     "single vertex!")
                tan_in = prev_vertex - first_vertex
                # often CLOSEPOLY is used when the curve has already reached
                # it's initial point in order to prevent there from being a
                # stray straight line segment (like closing a circle).
                # If it's used this way, then we more or less ignore the
                # current bcurve.
                if np.isclose(np.linalg.norm(tan_in), 0):
                    incidence_a, corner_a = _vertex_info_from_angles(
                            prev_tan_angle, first_tan_angle)
                    yield VertexInfo(prev_vertex, incidence_a, corner_a)
                    continue
                # otherwise, we have to calculate both the corner from the
                # previous line segment to the current straight line, and from
                # the current straight line to the original starting line. The
                # former is taken care of by the non-special-case code below.
                # the latter looks like:
                tan_out = B.tan_out
                angle_end = np.arctan2(tan_out[1], tan_out[0])
                incidence_a, corner_a = _vertex_info_from_angles(
                        angle_end, first_tan_angle)
                yield VertexInfo(first_vertex, incidence_a, corner_a)
            # finally, usual case is when two curves meet at an angle
            tan_in = -B.tan_in
            angle_in = np.arctan2(tan_in[1], tan_in[0])
            if first_tan_angle is None:
                first_tan_angle = angle_in
            if prev_tan_angle is not None:
                incidence_a, corner_a = _vertex_info_from_angles(
                        angle_in, prev_tan_angle)
                yield VertexInfo(prev_vertex, incidence_a, corner_a)
            tan_out = B.tan_out
            prev_tan_angle = np.arctan2(tan_out[1], tan_out[0])
            prev_vertex = B.control_points[-1]
        if prev_tan_angle is not None and is_capped:
            for cap_angle, cap_vertex in [(first_tan_angle, first_vertex),
                                          (prev_tan_angle, prev_vertex)]:
                yield VertexInfo(cap_vertex, cap_angle, None)

    @cbook._delete_parameter("3.3", "quantize")
    def cleaned(self, transform=None, remove_nans=False, clip=None,
                quantize=False, simplify=False, curves=False,
                stroke_width=1.0, snap=False, sketch=None):
        """
        Return a new Path with vertices and codes cleaned according to the
        parameters.

        See Also
        --------
        Path.iter_segments : for details of the keyword arguments.
        """
        vertices, codes = _path.cleanup_path(
            self, transform, remove_nans, clip, snap, stroke_width, simplify,
            curves, sketch)
        pth = Path._fast_from_codes_and_verts(vertices, codes, self)
        if not simplify:
            pth._should_simplify = False
        return pth

    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        See Also
        --------
        matplotlib.transforms.TransformedPath
            A specialized path class that will cache the transformed result and
            automatically update when the transform changes.
        """
        return Path(transform.transform(self.vertices), self.codes,
                    self._interpolation_steps)

    def contains_point(self, point, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given point.

        The path is always treated as closed; i.e. if the last code is not
        CLOSEPOLY an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check.
        transform : `matplotlib.transforms.Transform`, optional
            If not ``None``, *point* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *point*.
        radius : float, default: 0
            Add an additional margin on the path in coordinates of *point*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        bool

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        if transform is not None:
            transform = transform.frozen()
        # `point_in_path` does not handle nonlinear transforms, so we
        # transform the path ourselves.  If *transform* is affine, letting
        # `point_in_path` handle the transform avoids allocating an extra
        # buffer.
        if transform and not transform.is_affine:
            self = transform.transform_path(self)
            transform = None
        return _path.point_in_path(point[0], point[1], radius, self, transform)

    def contains_points(self, points, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given points.

        The path is always treated as closed; i.e. if the last code is not
        CLOSEPOLY an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0
            Add an additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        if transform is not None:
            transform = transform.frozen()
        result = _path.points_in_path(points, radius, self, transform)
        return result.astype('bool')

    def contains_path(self, path, transform=None):
        """
        Return whether this (closed) path completely contains the given path.

        If *transform* is not ``None``, the path will be transformed before
        checking for containment.
        """
        if transform is not None:
            transform = transform.frozen()
        return _path.path_in_path(self, None, path, transform)

    def get_extents(self, transform=None, **kwargs):
        """
        Get Bbox of the path.

        Parameters
        ----------
        transform : matplotlib.transforms.Transform, optional
            Transform to apply to path before computing extents, if any.
        **kwargs
            Forwarded to `.iter_bezier`.

        Returns
        -------
        matplotlib.transforms.Bbox
            The extents of the path Bbox([[xmin, ymin], [xmax, ymax]])
        """
        from .transforms import Bbox
        if transform is not None:
            self = transform.transform_path(self)
        if self.codes is None:
            xys = self.vertices
        elif len(np.intersect1d(self.codes, [Path.CURVE3, Path.CURVE4])) == 0:
            xys = self.vertices[self.codes != Path.CLOSEPOLY]
        else:
            xys = []
            for curve, code in self.iter_bezier(**kwargs):
                # places where the derivative is zero can be extrema
                _, dzeros = curve.axis_aligned_extrema()
                # as can the ends of the curve
                xys.append(curve([0, *dzeros, 1]))
            xys = np.concatenate(xys)
        if len(xys):
            return Bbox([xys.min(axis=0), xys.max(axis=0)])
        else:
            return Bbox.null()

    def get_stroked_extents(self, markeredgewidth, transform, joinstyle,
                            capstyle, **kwargs):
        """
        Get Bbox of path stroked with given *markeredgewidth*.

        Parameters
        ----------
        markeredgewidth : float
            Width, in points, of the stroke used to create the marker's edge.
            For ``markeredgewidth = 0``, same as `.get_extents`.
        transform : `~.transforms.Transform`
            Transform from the coordinates of the path's vertices to the units
            in which the marker edge is specified. The *markeredgewidth* is
            typically defined in points, so it doesn't usually make sense to
            request the stroked extents of a path without transforming it.
        joinstyle : {'miter', 'bevel', 'round'}
            How line corners are to be drawn.
        capstyle : {'butt', 'round', 'projecting'}
            How line ends are to be drawn.
        **kwargs
            Forwarded to `.iter_angles`.

        Returns
        -------
        bbox : (4,) float, array_like
            The extents of the path including an edge of width
            ``markeredgewidth``.

        Note
        ----
        The approach used is simply to notice that the bbox with no marker edge
        must be defined by a corner (control point of the linear parts of path)
        or an extremal point on one of the curved parts of the path.

        For a nonzero marker edge width, because the interior extrema will by
        definition be parallel to the bounding box, we need only check if the
        path location + width/2 extends the bbox at each interior extrema.
        Then, for each join and cap, we check if that join extends the bbox.
        """
        from .transforms import Bbox
        maxi = 2  # [xmin, ymin, *xmax, ymax]
        # get_extents returns a bbox, Bbox.extents returns a copy of a np.array
        extents = self.get_extents(transform=transform).extents
        for vinfo in self.iter_angles(transform=transform, **kwargs):
            _pad_extents_stroked_vertex(extents, vinfo, markeredgewidth,
                                        joinstyle, capstyle)
            # account for 2-fold ambiguity in orientation of corner's bisector
            # angle when the line is approximately straight (corner_angle = pi)
            corner_a = vinfo.corner_angle
            if corner_a is not None and np.isclose(vinfo.corner_angle, np.pi):
                # rotate by pi, this is the "same" corner, but padding in
                # opposite direction
                x = np.cos(vinfo.incidence_angle)
                y = np.sin(vinfo.incidence_angle)
                vinfo = VertexInfo(vinfo.apex, np.arctan2(-y, -x), np.pi)
                _pad_extents_stroked_vertex(extents, vinfo, markeredgewidth,
                                            joinstyle, capstyle)
        return Bbox.from_extents(extents)

    def intersects_path(self, other, filled=True):
        """
        Return whether if this path intersects another given path.

        If *filled* is True, then this also returns True if one path completely
        encloses the other (i.e., the paths are treated as filled).
        """
        return _path.path_intersects_path(self, other, filled)

    def intersects_bbox(self, bbox, filled=True):
        """
        Return whether this path intersects a given `~.transforms.Bbox`.

        If *filled* is True, then this also returns True if the path completely
        encloses the `.Bbox` (i.e., the path is treated as filled).

        The bounding box is always considered filled.
        """
        return _path.path_intersects_rectangle(
            self, bbox.x0, bbox.y0, bbox.x1, bbox.y1, filled)

    def interpolated(self, steps):
        """
        Return a new path resampled to length N x steps.

        Codes other than LINETO are not handled correctly.
        """
        if steps == 1:
            return self

        vertices = simple_linear_interpolation(self.vertices, steps)
        codes = self.codes
        if codes is not None:
            new_codes = np.full((len(codes) - 1) * steps + 1, Path.LINETO,
                                dtype=self.code_type)
            new_codes[0::steps] = codes
        else:
            new_codes = None
        return Path(vertices, new_codes)

    def to_polygons(self, transform=None, width=0, height=0, closed_only=True):
        """
        Convert this path to a list of polygons or polylines.  Each
        polygon/polyline is an Nx2 array of vertices.  In other words,
        each polygon has no ``MOVETO`` instructions or curves.  This
        is useful for displaying in backends that do not support
        compound paths or Bezier curves.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.

        If *closed_only* is `True` (default), only closed polygons,
        with the last point being the same as the first point, will be
        returned.  Any unclosed polylines in the path will be
        explicitly closed.  If *closed_only* is `False`, any unclosed
        polygons in the path will be returned as unclosed polygons,
        and the closed polygons will be returned explicitly closed by
        setting the last point to the same as the first point.
        """
        if len(self.vertices) == 0:
            return []

        if transform is not None:
            transform = transform.frozen()

        if self.codes is None and (width == 0 or height == 0):
            vertices = self.vertices
            if closed_only:
                if len(vertices) < 3:
                    return []
                elif np.any(vertices[0] != vertices[-1]):
                    vertices = [*vertices, vertices[0]]

            if transform is None:
                return [vertices]
            else:
                return [transform.transform(vertices)]

        # Deal with the case where there are curves and/or multiple
        # subpaths (using extension code)
        return _path.convert_path_to_polygons(
            self, transform, width, height, closed_only)

    _unit_rectangle = None

    @classmethod
    def unit_rectangle(cls):
        """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
        if cls._unit_rectangle is None:
            cls._unit_rectangle = cls([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                                      closed=True, readonly=True)
        return cls._unit_rectangle

    _unit_regular_polygons = WeakValueDictionary()

    @classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        Return a :class:`Path` instance for a unit regular polygon with the
        given *numVertices* such that the circumscribing circle has radius 1.0,
        centered at (0, 0).
        """
        if numVertices <= 16:
            path = cls._unit_regular_polygons.get(numVertices)
        else:
            path = None
        if path is None:
            theta = ((2 * np.pi / numVertices) * np.arange(numVertices + 1)
                     # This initial rotation is to make sure the polygon always
                     # "points-up".
                     + np.pi / 2)
            verts = np.column_stack((np.cos(theta), np.sin(theta)))
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_polygons[numVertices] = path
        return path

    _unit_regular_stars = WeakValueDictionary()

    @classmethod
    def unit_regular_star(cls, numVertices, innerCircle=0.5):
        """
        Return a :class:`Path` for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        if numVertices <= 16:
            path = cls._unit_regular_stars.get((numVertices, innerCircle))
        else:
            path = None
        if path is None:
            ns2 = numVertices * 2
            theta = (2*np.pi/ns2 * np.arange(ns2 + 1))
            # This initial rotation is to make sure the polygon always
            # "points-up"
            theta += np.pi / 2.0
            r = np.ones(ns2 + 1)
            r[1::2] = innerCircle
            verts = (r * np.vstack((np.cos(theta), np.sin(theta)))).T
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_stars[(numVertices, innerCircle)] = path
        return path

    @classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        Return a :class:`Path` for a unit regular asterisk with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        return cls.unit_regular_star(numVertices, 0.0)

    _unit_circle = None

    @classmethod
    def unit_circle(cls):
        """
        Return the readonly :class:`Path` of the unit circle.

        For most cases, :func:`Path.circle` will be what you want.
        """
        if cls._unit_circle is None:
            cls._unit_circle = cls.circle(center=(0, 0), radius=1,
                                          readonly=True)
        return cls._unit_circle

    @classmethod
    def circle(cls, center=(0., 0.), radius=1., readonly=False):
        """
        Return a `Path` representing a circle of a given radius and center.

        Parameters
        ----------
        center : (float, float), default: (0, 0)
            The center of the circle.
        radius : float, default: 1
            The radius of the circle.
        readonly : bool
            Whether the created path should have the "readonly" argument
            set when creating the Path instance.

        Notes
        -----
        The circle is approximated using 8 cubic Bezier curves, as described in

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <https://www.tinaja.com/glib/ellipse4.pdf>`_.
        """
        MAGIC = 0.2652031
        SQRTHALF = np.sqrt(0.5)
        MAGIC45 = SQRTHALF * MAGIC

        vertices = np.array([[0.0, -1.0],

                             [MAGIC, -1.0],
                             [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                             [SQRTHALF, -SQRTHALF],

                             [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                             [1.0, -MAGIC],
                             [1.0, 0.0],

                             [1.0, MAGIC],
                             [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                             [SQRTHALF, SQRTHALF],

                             [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                             [MAGIC, 1.0],
                             [0.0, 1.0],

                             [-MAGIC, 1.0],
                             [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
                             [-SQRTHALF, SQRTHALF],

                             [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
                             [-1.0, MAGIC],
                             [-1.0, 0.0],

                             [-1.0, -MAGIC],
                             [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
                             [-SQRTHALF, -SQRTHALF],

                             [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
                             [-MAGIC, -1.0],
                             [0.0, -1.0],

                             [0.0, -1.0]],
                            dtype=float)

        codes = [cls.CURVE4] * 26
        codes[0] = cls.MOVETO
        codes[-1] = cls.CLOSEPOLY
        return Path(vertices * radius + center, codes, readonly=readonly)

    _unit_circle_righthalf = None

    @classmethod
    def unit_circle_righthalf(cls):
        """
        Return a `Path` of the right half of a unit circle.

        See `Path.circle` for the reference on the approximation used.
        """
        if cls._unit_circle_righthalf is None:
            MAGIC = 0.2652031
            SQRTHALF = np.sqrt(0.5)
            MAGIC45 = SQRTHALF * MAGIC

            vertices = np.array(
                [[0.0, -1.0],

                 [MAGIC, -1.0],
                 [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                 [SQRTHALF, -SQRTHALF],

                 [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                 [1.0, -MAGIC],
                 [1.0, 0.0],

                 [1.0, MAGIC],
                 [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                 [SQRTHALF, SQRTHALF],

                 [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                 [MAGIC, 1.0],
                 [0.0, 1.0],

                 [0.0, -1.0]],

                float)

            codes = np.full(14, cls.CURVE4, dtype=cls.code_type)
            codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

            cls._unit_circle_righthalf = cls(vertices, codes, readonly=True)
        return cls._unit_circle_righthalf

    @classmethod
    def arc(cls, theta1, theta2, n=None, is_wedge=False):
        """
        Return the unit circle arc from angles *theta1* to *theta2* (in
        degrees).

        *theta2* is unwrapped to produce the shortest arc within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
        *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

           Masionobe, L.  2003.  `Drawing an elliptical arc using
           polylines, quadratic or cubic Bezier curves
           <http://www.spaceroots.org/documents/ellipse/index.html>`_.
        """
        halfpi = np.pi * 0.5

        eta1 = theta1
        eta2 = theta2 - 360 * np.floor((theta2 - theta1) / 360)
        # Ensure 2pi range is not flattened to 0 due to floating-point errors,
        # but don't try to expand existing 0 range.
        if theta2 != theta1 and eta2 <= eta1:
            eta2 += 360
        eta1, eta2 = np.deg2rad([eta1, eta2])

        # number of curve segments to make
        if n is None:
            n = int(2 ** np.ceil((eta2 - eta1) / halfpi))
        if n < 1:
            raise ValueError("n must be >= 1 or None")

        deta = (eta2 - eta1) / n
        t = np.tan(0.5 * deta)
        alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0

        steps = np.linspace(eta1, eta2, n + 1, True)
        cos_eta = np.cos(steps)
        sin_eta = np.sin(steps)

        xA = cos_eta[:-1]
        yA = sin_eta[:-1]
        xA_dot = -yA
        yA_dot = xA

        xB = cos_eta[1:]
        yB = sin_eta[1:]
        xB_dot = -yB
        yB_dot = xB

        if is_wedge:
            length = n * 3 + 4
            vertices = np.zeros((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            vertices[1] = [xA[0], yA[0]]
            codes[0:2] = [cls.MOVETO, cls.LINETO]
            codes[-2:] = [cls.LINETO, cls.CLOSEPOLY]
            vertex_offset = 2
            end = length - 2
        else:
            length = n * 3 + 1
            vertices = np.empty((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            vertices[0] = [xA[0], yA[0]]
            codes[0] = cls.MOVETO
            vertex_offset = 1
            end = length

        vertices[vertex_offset:end:3, 0] = xA + alpha * xA_dot
        vertices[vertex_offset:end:3, 1] = yA + alpha * yA_dot
        vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
        vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
        vertices[vertex_offset+2:end:3, 0] = xB
        vertices[vertex_offset+2:end:3, 1] = yB

        return cls(vertices, codes, readonly=True)

    @classmethod
    def wedge(cls, theta1, theta2, n=None):
        """
        Return the unit circle wedge from angles *theta1* to *theta2* (in
        degrees).

        *theta2* is unwrapped to produce the shortest wedge within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*
        to *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

        See `Path.arc` for the reference on the approximation used.
        """
        return cls.arc(theta1, theta2, n, True)

    @staticmethod
    @lru_cache(8)
    def hatch(hatchpattern, density=6):
        """
        Given a hatch specifier, *hatchpattern*, generates a Path that
        can be used in a repeated hatching pattern.  *density* is the
        number of lines per unit square.
        """
        from matplotlib.hatch import get_path
        return (get_path(hatchpattern, density)
                if hatchpattern is not None else None)

    def clip_to_bbox(self, bbox, inside=True):
        """
        Clip the path to the given bounding box.

        The path must be made up of one or more closed polygons.  This
        algorithm will not behave correctly for unclosed paths.

        If *inside* is `True`, clip to the inside of the box, otherwise
        to the outside of the box.
        """
        # Use make_compound_path_from_polys
        verts = _path.clip_path_to_rect(self, bbox, inside)
        paths = [Path(poly) for poly in verts]
        return self.make_compound_path(*paths)


def get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_transform):
    r"""
    Given a sequence of `Path`\s, `~.Transform`\s objects, and offsets, as
    found in a `~.PathCollection`, returns the bounding box that encapsulates
    all of them.

    Parameters
    ----------
    master_transform : `~.Transform`
        Global transformation applied to all paths.
    paths : list of `Path`
    transforms : list of `~.Affine2D`
    offsets : (N, 2) array-like
    offset_transform : `~.Affine2D`
        Transform applied to the offsets before offsetting the path.

    Notes
    -----
    The way that *paths*, *transforms* and *offsets* are combined
    follows the same method as for collections:  Each is iterated over
    independently, so if you have 3 paths, 2 transforms and 1 offset,
    their combinations are as follows:

        (A, A, A), (B, B, A), (C, A, A)
    """
    from .transforms import Bbox
    if len(paths) == 0:
        raise ValueError("No paths provided")
    return Bbox.from_extents(*_path.get_path_collection_extents(
        master_transform, paths, np.atleast_3d(transforms),
        offsets, offset_transform))


def _vertex_info_from_angles(angle_1, angle_2):
    r"""
    Gets `.VertexInfo` from direction of lines making up a corner.

    Helper function for `.iter_angles`.

    Parameters
    ----------
    angle_1 : float in [-pi, pi]
        Orientation of the tangent vector of the first line at the corner, if
        the tangent vector is defined to be oriented pointing along the line
        towards the corner's apex. See `Examples` for clarification.
    angle_2 : float
        Same as *angle_1* but for the other line forming the corner.

    Returns
    -------
    incidence_angle : float in [-pi, pi]
        as described in `.VertexInfo` docs
    corner_angle : float in [0, pi]
        as described in `.VertexInfo` docs

    Notes
    -----
    Is necessarily ambiguous if corner_angle is pi.

    Examples
    --------
    The only corner in ``Path([[0, 0], [1, 1], [0, 1]])`` would have two lines,
    (``Path([[0, 0], [1, 1]])`` and ``Path([[1, 1], [0, 1]])``), with angles
    :math:`\pi/4` and :math:`0`, respectively.
    """
    # get "interior" angle between tangents to joined curves' tips
    corner_angle = np.abs(angle_1 - angle_2)
    if corner_angle > np.pi:
        corner_angle = 2*np.pi - corner_angle
    # since input in [-pi, pi], we need to sort to avoid a modulo op
    smaller_angle = min(angle_1, angle_2)
    larger_angle = max(angle_1, angle_2)
    if np.isclose(smaller_angle + corner_angle, larger_angle):
        incidence_angle = smaller_angle + corner_angle/2
    else:
        incidence_angle = smaller_angle - corner_angle/2
        # stay in [-pi, pi]
        if incidence_angle < -np.pi:
            incidence_angle = 2*np.pi + incidence_angle
    return incidence_angle, corner_angle


def _stroke_x_overflow(width, phi, theta, joinstyle='miter', capstyle='butt'):
    r"""
    Compute distance a stroke of *width* extends past x coordinate of vertex.

    The formulas here only work for corners which "point" in the positive x
    direction. That is, we assume the incident lines are both coming from the
    left, and extension is only measured to the right.

    Parameters
    ----------
    width : float
        `markeredgewidth` used to draw the stroke that we're computing the
        overflow for.
    phi : float
        For vertices with one incoming line, *phi* is the incidence angle that
        line forms with the positive y-axis.  For corners (vertices with two
        incoming lines) the incidence angle of the corner's bisector is used
        instead.
    theta : float or None
        For vertices with one incoming line, set to ``None``. For vertices that
        form a corner, the interior angle swept out by the two lines that meet
        at the vertex.
    joinstyle : {'miter', 'bevel', 'round'}
        How line corners are to be drawn.
    capstyle : {'butt', 'round', 'projecting'}
        How line ends are to be drawn.

    Returns
    -------
    pad : float
        Amount of bbox overflow.

    Notes
    -----
    $\pi$ and 0 are equivalent for *theta*. Both $\phi$ and $\pi - \phi$
    are equivalent for *phi* by symmetry.

    If and only if `.VertexInfo.incidence_angle` is in :math:`[-\pi, \pi]`,
    then *phi* corresponds to `.VertexInfo.incidence_angle`. *theta* always
    corresponds to `.VertexInfo.corner_angle`.
    """
    if theta is not None and (theta < 0 or theta > np.pi) \
            or phi < 0 or phi > np.pi:
        raise ValueError("Corner angles should be in [0, pi].")
    if phi > np.pi/2:
        # equivalent by symmetry, but keeps math simpler
        phi = np.pi - phi
    # if there's no corner (i.e. the path just ends, as in the "sides" of the
    # caret marker (and other non-fillable markers), we still need to compute
    # how much the "cap" extends past the endpoint of the path
    if theta is None:
        # for "butt" caps we can compute how far the
        # outside edge of the markeredge stroke extends outside of the bounding
        # box of its path using the law of sines: $\sin(\phi)/(w/2) =
        # \sin(\pi/2 - \phi)/l$ for $w$ the `markeredgewidth`, $\phi$ the
        # incidence angle of the line, then $l$ is the length along the outer
        # edge of the stroke that extends beyond the bouding box. We can
        # translate this to a distance perpendicular to the bounding box E(w,
        # \phi) = l \sin(\phi)$, for $l$ as above.
        if capstyle == 'butt':
            return (width/2) * np.cos(phi)
        # "round" caps are hemispherical, so regardless of angle
        elif capstyle == 'round':
            return width/2
        # finally, projecting caps are just bevel caps with an extra
        # width/2 distance along the direction of the line, so "butt" plus some
        # extra
        elif capstyle == 'projecting':
            return (width/2) * np.cos(phi) + (width/2)*np.sin(phi)
        else:
            raise ValueError(f"Unknown capstyle: {capstyle}.")
    # the two "same as straight line" cases are NaN limits in the miter formula
    elif (np.isclose(theta, 0) and np.isclose(phi, 0)
          or np.isclose(theta, np.pi) and np.isclose(phi, np.pi/2)):
        return width/2
    # to calculate the offset for _joinstyle == 'miter', imagine aligning the
    # corner so that one line comes in along the negative x-axis, and another
    # from above, making an angle $\theta$ with the negative x-axis. The tip of
    # the new corner created by the markeredge stroke will be at the point
    # where the two outer edge of the markeredge stroke intersect.  in the
    # orientation described above, the outer edge of the stroke aligned with
    # the x-axis will obviously have equation $y = -w/2$ where $w$ is the
    # markeredgewidth. WLOG, the stroke coming in from above at an angle
    # $\theta$ from the negative x-axis will have equation
    # $$-(\tan(\theta) x + \frac{w}{2\cos(\theta)}.$$
    # the intersection of these two lines is at $y = w/2$, and we can solve for
    # $x = \cot(\theta) (\frac{w}{2} + \frac{w}{2\cos(\theta)})$.
    # this puts the "edge" tip a distance $M = (w/2)\csc(\theta/2)$
    # from the tip of the corner itself, on the line defined by the bisector of
    # the corner angle. So the extra padding required is $M\sin(\phi)$, where
    # $\phi$ is the incidence angle of the corner's bisector. Notice that in
    # the limit ($\phi = \theta/2$) where the "corner" is flush with the bbox,
    # this correctly simplifies to just $w/2$.
    elif joinstyle == 'miter':
        # matplotlib currently doesn't set the miterlimit...
        _nominal_miter_limit = 10  # pdf and agg do this
        if 1/np.sin(theta/2) > _nominal_miter_limit:
            return _stroke_x_overflow(width, phi, theta, 'bevel', capstyle)
        else:
            return (width/2)*np.sin(phi)/np.sin(theta/2)
    # a beveled edge is exactly the convex hull of its two composite lines with
    # capstyle='butt'. So we just compute the individual lines' incidence
    # angles and take the maximum of the two padding values
    elif joinstyle == 'bevel':
        phi1 = phi + theta/2
        phi2 = phi - theta/2
        return (width/2) * max(np.abs(np.cos(phi1)), np.abs(np.cos(phi2)))
    # finally, _joinstyle = "round" is just _joinstyle = "bevel" but with
    # a hemispherical cap. we could calculate this but for now no markers use
    # it....except those with "no corner", in which case we can treat them the
    # same as squares...
    elif joinstyle == 'round':
        return width/2  # hemispherical cap, so always same padding
    else:
        raise ValueError(f"Unknown joinstyle: {joinstyle}")


def _pad_extents_stroked_vertex(extents, vinfo, markeredgewidth, joinstyle,
                                capstyle):
    """
    Accumulator for building true extents from `.VertexInfo`s.

    Parameters
    ----------
    extents : (4,) array_like of float
        The extents (xmin, ymin, xmax, ymax) of the `~.transforms.Bbox` of the
        vertices. Modified in place so that the corner described by *vinfo*
        fits into the extents when stroked with a width of *markeredgewidth*.
    vinfo : `.VertexInfo`
        Information about the corner or cap at one vertex.
    markeredgewidth : `float`
        The width of the stroke being drawn.
    joinstyle : {'miter', 'bevel', 'round'}
        How line corners are to be drawn.
    capstyle : {'butt', 'round', 'projecting'}
        How line ends are to be drawn.

    Notes
    -----
    Implemented by wrapping `._stroke_x_overflow`. This function checks which
    direction the corner (or cap) is pointing, then for each side of *extents*
    that might need padding, it rotates the corner to point in the positive x
    direction and calls `._stroke_x_overflow` to get the padding.
    """
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3
    # now for each direction (up/down/left/right), convert the absolute
    # incidence angle into the incidence angle relative to that respective side
    # of the bbox, and see if the stroked vertex expands the extents...
    x, y = vinfo.apex
    if np.cos(vinfo.incidence_angle) > 0:
        incidence_angle = vinfo.incidence_angle + np.pi/2
        x += _stroke_x_overflow(markeredgewidth, incidence_angle,
                                vinfo.corner_angle, joinstyle, capstyle)
        if x > extents[xmax]:
            extents[xmax] = x
    else:
        if vinfo.incidence_angle < 0:  # [-pi, -pi/2]
            incidence_angle = 3*np.pi/2 + vinfo.incidence_angle
        else:
            incidence_angle = vinfo.incidence_angle - np.pi/2
        x -= _stroke_x_overflow(markeredgewidth, incidence_angle,
                                vinfo.corner_angle, joinstyle, capstyle)
        if x < extents[xmin]:
            extents[xmin] = x
    if np.sin(vinfo.incidence_angle) > 0:
        incidence_angle = vinfo.incidence_angle
        y += _stroke_x_overflow(markeredgewidth, incidence_angle,
                                vinfo.corner_angle, joinstyle, capstyle)
        if y > extents[ymax]:
            extents[ymax] = y
    else:
        incidence_angle = vinfo.incidence_angle + np.pi
        y -= _stroke_x_overflow(markeredgewidth, incidence_angle,
                                vinfo.corner_angle, joinstyle, capstyle)
        if y < extents[ymin]:
            extents[ymin] = y
    # also catch extra extent due to caps growing sideways
    if vinfo.corner_angle is None:
        for perp_dir in [np.pi/2, 3*np.pi/2]:
            x, y = vinfo.apex
            cap_perp = vinfo.incidence_angle + perp_dir
            x += (markeredgewidth/2) * np.cos(cap_perp)
            if x < extents[xmin]:
                extents[xmin] = x
            if x > extents[xmax]:
                extents[xmax] = x
            y += (markeredgewidth/2) * np.sin(cap_perp)
            if y < extents[ymin]:
                extents[ymin] = y
            if y > extents[ymax]:
                extents[ymax] = y
