r"""
A module for dealing with the polylines used throughout Matplotlib.

The primary class for polyline handling in Matplotlib is `Path`.  Almost all
vector drawing makes use of `Path`\s somewhere in the drawing pipeline.

Whilst a `Path` instance itself cannot be drawn, some `.Artist` subclasses,
such as `.PathPatch` and `.PathCollection`, can be used for convenient `Path`
visualisation.
"""

from functools import lru_cache
from weakref import WeakValueDictionary

import numpy as np

import matplotlib as mpl
from . import _path, cbook
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment, split_bezier_intersecting_with_closedpath


def _update_extents(extents, point):
    dim = len(point)
    for i, xi in enumerate(point):
        if xi < extents[i]:
            extents[i] = xi
        # elif here would fail to correctly update from "null" extents of
        # np.array([np.inf, np.inf, -np.inf, -np.inf])
        if extents[i+dim] < xi:
            extents[i+dim] = xi


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
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError(
                "'vertices' must be a 2D list or array with shape Nx2")

        if codes is not None:
            codes = np.asarray(codes, self.code_type)
            if codes.ndim != 1 or len(codes) != len(vertices):
                raise ValueError("'codes' must be a 1D list or array with the "
                                 "same length of 'vertices'")
            if len(codes) and codes[0] != self.MOVETO:
                raise ValueError("The first element of 'code' must be equal "
                                 "to 'MOVETO' ({})".format(self.MOVETO))
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
        The list of codes in the `Path` as a 1-D numpy array.  Each
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

    @cbook.deprecated(
        "3.1", alternative="not np.isfinite(self.vertices).all()")
    @property
    def has_nonfinite(self):
        """
        `True` if the vertices array has nonfinite values.
        """
        return not np.isfinite(self._vertices).all()

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
        Iterates over all of the curve segments in the path.  Each iteration
        returns a 2-tuple ``(vertices, code)``, where ``vertices`` is a
        sequence of 1-3 coordinate pairs, and ``code`` is a `Path` code.

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
            If None, perform snapping if the path contains only segments
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
        kwargs : Dict[str, object]
            Forwareded to iter_segments.

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
        for vertices, code in self.iter_segments(**kwargs):
            if first_vert is None:
                if code != Path.MOVETO:
                    raise ValueError("Malformed path, must start with MOVETO.")
            if code == Path.MOVETO:  # a point is like "CURVE1"
                first_vert = vertices
                yield BezierSegment(np.array([first_vert])), code
            elif code == Path.LINETO:  # "CURVE2"
                yield BezierSegment(np.array([prev_vert, vertices])), code
            elif code == Path.CURVE3:
                yield BezierSegment(np.array([prev_vert, vertices[:2],
                                    vertices[2:]])), code
            elif code == Path.CURVE4:
                yield BezierSegment(np.array([prev_vert, vertices[:2],
                                    vertices[2:4], vertices[4:]])), code
            elif code == Path.CLOSEPOLY:
                yield BezierSegment(np.array([prev_vert, first_vert])), code
            elif code == Path.STOP:
                return
            else:
                raise ValueError("Invalid Path.code_type: " + str(code))
            prev_vert = vertices[-2:]

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
        Return whether the (closed) path contains the given point.

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
        Return whether the (closed) path contains the given point.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0.
            Add an additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array
        """
        if transform is not None:
            transform = transform.frozen()
        result = _path.points_in_path(points, radius, self, transform)
        return result.astype('bool')

    def contains_path(self, path, transform=None):
        """
        Return whether this (closed) path completely contains the given path.

        If *transform* is not ``None``, the path will be transformed before
        performing the test.
        """
        if transform is not None:
            transform = transform.frozen()
        return _path.path_in_path(self, None, path, transform)

    def get_extents(self, transform=None):
        """
        Return the extents (*xmin*, *ymin*, *xmax*, *ymax*) of the path.

        Unlike computing the extents on the *vertices* alone, this
        algorithm will take into account the curves and deal with
        control points appropriately.
        """
        from .transforms import Bbox
        path = self
        if transform is not None:
            transform = transform.frozen()
            if not transform.is_affine:
                path = self.transformed(transform)
                transform = None
        return Bbox(_path.get_path_extents(path, transform))

    def get_exact_extents(self, **kwargs):
        """Get size of Bbox of curve (instead of Bbox of control points).

        Parameters
        ----------
        kwargs : Dict[str, object]
            Forwarded to self.iter_bezier.

        Returns
        -------
        extents : (4,) float, array_like
            The extents of the path (xmin, ymin, xmax, ymax).
        """
        maxi = 2  # [xmin, ymin, *xmax, ymax]
        # return value for empty paths to match _path.h
        extents = np.array([np.inf, np.inf, -np.inf, -np.inf])
        for curve, code in self.iter_bezier(**kwargs):
            # start and endpoints can be extrema of the curve
            _update_extents(extents, curve(0))  # start point
            _update_extents(extents, curve(1))  # end point
            # interior extrema where d/ds B(s) == 0
            _, dzeros = curve.axis_aligned_extrema
            if len(dzeros) == 0:
                continue
            for zero in dzeros:
                potential_extrema = curve.point_at_t(zero)
                _update_extents(extents, potential_extrema)
        return extents

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

    def length(self, rtol=None, atol=None, **kwargs):
        """Get length of Path.

        Equivalent to (but not computed as)

        .. math::

            \sum_{j=1}^N \int_0^1 ||B'_j(t)|| dt

        where the sum is over the :math:`N` Bezier curves that comprise the
        Path. Notice that this measure of length will assign zero weight to all
        isolated points on the Path.

        Returns
        -------
        length : float
            The path length.
        """
        return np.sum([B.arc_length(rtol, atol)
                       for B, code in self.iter_bezier(**kwargs)])

    def signed_area(self, **kwargs):
        """
        Get signed area filled by path.

        All sub paths are treated as if they had been closed. That is, if there
        is a MOVETO without a preceding CLOSEPOLY, one is added.

        Signed area means that if a path is self-intersecting, the drawing rule
        "even-odd" is used and only the filled area is counted.

        Returns
        -------
        area : float
            The (signed) enclosed area of the path.
        """
        area = 0
        prev_point = None
        prev_code = None
        start_point = None
        for B, code in self.iter_bezier(**kwargs):
            if code == Path.MOVETO:
                if prev_code is not None and prev_code is not Path.CLOSEPOLY:
                    Bclose = BezierSegment(np.array([prev_point, start_point]))
                    area += Bclose.arc_area()
                start_point = B.control_points[0]
            area += B.arc_area()
            prev_point = B.control_points[-1]
            prev_code = code
        # add final implied CLOSEPOLY, if necessary
        if start_point is not None \
                and not np.all(np.isclose(start_point, prev_point)):
            Bclose = BezierSegment(np.array([prev_point, start_point]))
            area += Bclose.arc_area()
        return area

    def center_of_mass(self, dimension=None, **kwargs):
        r"""
        Center of mass of the path, assuming constant density.

        The center of mass is defined to be the expected value of a vector
        located uniformly within either the filled area of the path
        (:code:`dimension=2`) or the along path's edge (:code:`dimension=1`) or
        along isolated points of the path (:code:`dimension=0`).  Notice in
        particular that for this definition, if the filled area is used, then
        any 0- or 1-dimensional components of the path will not contribute to
        the center of mass. Similarly, for if *dimension* is 1, then isolated
        points in the path (i.e.  "0-dimensional" strokes made up of only
        :code:`Path.MOVETO`'s) will not contribute to the center of mass.

        For the 2d case, the center of mass is computed using the same
        filling strategy as `signed_area`. So, if a path is self-intersecting,
        the drawing rule "even-odd" is used and only the filled area is
        counted, and all sub paths are treated as if they had been closed. That
        is, if there is a MOVETO without a preceding CLOSEPOLY, one is added.

        For the 1d measure, the curve is averaged as-is (the implied CLOSEPOLY
        is not added).

        For the 0d measure, any non-isolated points are ignored.

        Parameters
        ----------
        dimension : 2, 1, or 0 (optional)
            Whether to compute the center of mass by taking the expected value
            of a position uniformly distributed within the filled path
            (2D-measure), the path's edge (1D-measure), or between the
            discrete, isolated points of the path (0D-measure), respectively.
            By default, the intended dimension of the path is inferred by
            checking first if `Path.signed_area` is non-zero (implying a
            *dimension* of 2), then if the `Path.arc_length` is non-zero
            (implying a *dimension* of 1), and finally falling back to the
            counting measure (*dimension* of 0).
        kwargs : Dict[str, object]
            Passed thru to `Path.cleaned` via `Path.iter_bezier`.

        Returns
        -------
        r_cm : (2,) np.array<float>
            The center of mass of the path.

        Raises
        ------
        ValueError
            An empty path has no well-defined center of mass.

            In addition, if a specific *dimension* is requested and that
            dimension is not well-defined, an error is raised. This can happen
            if::

                1) 2D expected value was requested but the path has zero area
                2) 1D expected value was requested but the path has only
                `Path.MOVETO` directives
                3) 0D expected value was requested but the path has NO
                subsequent `Path.MOVETO` directives.

            This error cannot be raised if the function is allowed to infer
            what *dimension* to use.
        """
        area = None
        cleaned = self.cleaned(**kwargs)
        move_codes = cleaned.codes == Path.MOVETO
        if len(cleaned.codes) == 0:
            raise ValueError("An empty path has no center of mass.")
        if dimension is None:
            dimension = 2
            area = cleaned.signed_area()
            if not np.isclose(area, 0):
                dimension -= 1
            if np.all(move_codes):
                dimension = 0
        if dimension == 2:
            # area computation can be expensive, make sure we don't repeat it
            if area is None:
                area = cleaned.signed_area()
            if np.isclose(area, 0):
                raise ValueError("2d expected value over empty area is "
                                 "ill-defined.")
            return cleaned._2d_center_of_mass(area)
        if dimension == 1:
            if np.all(move_codes):
                raise ValueError("1d expected value over empty arc-length is "
                                 "ill-defined.")
            return cleaned._1d_center_of_mass()
        if dimension == 0:
            adjacent_moves = (move_codes[1:] + move_codes[:-1]) == 2
            if len(move_codes) > 1 and not np.any(adjacent_moves):
                raise ValueError("0d expected value with no isolated points "
                                 "is ill-defined.")
            return cleaned._0d_center_of_mass()

    def _2d_center_of_mass(self, normalization=None):
        #TODO: refactor this and signed_area (and maybe others, with
        # close= parameter)?
        if normalization is None:
            normalization = self.signed_area()
        r_cm = np.zeros(2)
        prev_point = None
        prev_code = None
        start_point = None
        for B, code in self.iter_bezier():
            if code == Path.MOVETO:
                if prev_code is not None and prev_code is not Path.CLOSEPOLY:
                    Bclose = BezierSegment(np.array([prev_point, start_point]))
                    r_cm += Bclose.arc_center_of_mass()
                start_point = B.control_points[0]
            r_cm += B.arc_center_of_mass()
            prev_point = B.control_points[-1]
            prev_code = code
        # add final implied CLOSEPOLY, if necessary
        if start_point is not None \
                and not np.all(np.isclose(start_point, prev_point)):
            Bclose = BezierSegment(np.array([prev_point, start_point]))
            r_cm += Bclose.arc_center_of_mass()
        return r_cm / normalization

    def _1d_center_of_mass(self):
        r_cm = np.zeros(2)
        Bs = list(self.iter_bezier())
        arc_lengths = np.array([B.arc_length() for B in Bs])
        r_cms = np.array([B.center_of_mass() for B in Bs])
        total_length = np.sum(arc_lengths)
        return np.sum(r_cms*arc_lengths)/total_length

    def _0d_center_of_mass(self):
        move_verts = self.codes
        isolated_verts = move_verts.copy()
        if len(move_verts) > 1:
            isolated_verts[:-1] = (move_verts[:-1] + move_verts[1:]) == 2
            isolated_verts[-1] = move_verts[-1]
        num_verts = np.sum(isolated_verts)
        return np.sum(self.vertices[isolated_verts], axis=0)/num_verts

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

    def split_path_inout(self, inside, tolerance=0.01, reorder_inout=False):
        """
        Divide a path into two segments at the point where ``inside(x, y)``
        becomes False.
        """
        path_iter = self.iter_segments()

        ctl_points, command = next(path_iter)
        begin_inside = inside(ctl_points[-2:])  # true if begin point is inside

        ctl_points_old = ctl_points

        iold = 0
        i = 1

        for ctl_points, command in path_iter:
            iold = i
            i += len(ctl_points) // 2
            if inside(ctl_points[-2:]) != begin_inside:
                bezier_path = np.concatenate([ctl_points_old[-2:], ctl_points])
                break
            ctl_points_old = ctl_points
        else:
            raise ValueError("The path does not intersect with the patch")

        bp = bezier_path.reshape((-1, 2))
        left, right = split_bezier_intersecting_with_closedpath(
            bp, inside, tolerance)
        if len(left) == 2:
            codes_left = [Path.LINETO]
            codes_right = [Path.MOVETO, Path.LINETO]
        elif len(left) == 3:
            codes_left = [Path.CURVE3, Path.CURVE3]
            codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        elif len(left) == 4:
            codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
            codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        else:
            raise AssertionError("This should never be reached")

        verts_left = left[1:]
        verts_right = right[:]

        if self.codes is None:
            path_in = Path(np.concatenate([self.vertices[:i], verts_left]))
            path_out = Path(np.concatenate([verts_right, self.vertices[i:]]))

        else:
            path_in = Path(np.concatenate([self.vertices[:iold], verts_left]),
                           np.concatenate([self.codes[:iold], codes_left]))

            path_out = Path(np.concatenate([verts_right, self.vertices[i:]]),
                            np.concatenate([codes_right, self.codes[i:]]))

        if reorder_inout and not begin_inside:
            path_in, path_out = path_out, path_in

        return path_in, path_out

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
          Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.
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
