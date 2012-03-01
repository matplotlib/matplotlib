"""
Contains a class for managing paths (polylines).
"""

import math
from weakref import WeakValueDictionary

import numpy as np
from numpy import ma

from matplotlib._path import point_in_path, get_path_extents, \
    point_in_path_collection, get_path_collection_extents, \
    path_in_path, path_intersects_path, convert_path_to_polygons, \
    cleanup_path
from matplotlib.cbook import simple_linear_interpolation, maxdict
from matplotlib import rcParams

class Path(object):
    """
    :class:`Path` represents a series of possibly disconnected,
    possibly closed, line and curve segments.

    The underlying storage is made up of two parallel numpy arrays:
      - *vertices*: an Nx2 float array of vertices
      - *codes*: an N-length uint8 array of vertex types

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices as well as three codes ``CURVE3``.

    The code types are:

       - ``STOP``   :  1 vertex (ignored)
           A marker for the end of the entire path (currently not
           required and ignored)

       - ``MOVETO`` :  1 vertex
            Pick up the pen and move to the given vertex.

       - ``LINETO`` :  1 vertex
            Draw a line from the current position to the given vertex.

       - ``CURVE3`` :  1 control point, 1 endpoint
          Draw a quadratic Bezier curve from the current position,
          with the given control point, to the given end point.

       - ``CURVE4`` :  2 control points, 1 endpoint
          Draw a cubic Bezier curve from the current position, with
          the given control points, to the given end point.

       - ``CLOSEPOLY`` : 1 vertex (ignored)
          Draw a line segment to the start point of the current
          polyline.

    Users of Path objects should not access the vertices and codes
    arrays directly.  Instead, they should use :meth:`iter_segments`
    to get the vertex/code pairs.  This is important, since many
    :class:`Path` objects, as an optimization, do not store a *codes*
    at all, but have a default one provided for them by
    :meth:`iter_segments`.

    .. note::

        The vertices and codes arrays should be treated as
        immutable -- there are a number of optimizations and assumptions
        made up front in the constructor that will not change when the
        data changes.

    """

    # Path codes
    STOP      = 0    # 1 vertex
    MOVETO    = 1    # 1 vertex
    LINETO    = 2    # 1 vertex
    CURVE3    = 3    # 2 vertices
    CURVE4    = 4    # 3 vertices
    CLOSEPOLY = 0x4f # 1 vertex

    NUM_VERTICES = [1, 1, 1, 2,
                    3, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1]

    code_type = np.uint8

    def __init__(self, vertices, codes=None, _interpolation_steps=1, closed=False):
        """
        Create a new path with the given vertices and codes.

        *vertices* is an Nx2 numpy float array, masked array or Python
        sequence.

        *codes* is an N-length numpy array or Python sequence of type
        :attr:`matplotlib.path.Path.code_type`.

        These two arrays must have the same length in the first
        dimension.

        If *codes* is None, *vertices* will be treated as a series of
        line segments.

        If *vertices* contains masked values, they will be converted
        to NaNs which are then handled correctly by the Agg
        PathIterator and other consumers of path data, such as
        :meth:`iter_segments`.

        *interpolation_steps* is used as a hint to certain projections,
        such as Polar, that this path should be linearly interpolated
        immediately before drawing.  This attribute is primarily an
        implementation detail and is not intended for public use.
        """
        if ma.isMaskedArray(vertices):
            vertices = vertices.astype(np.float_).filled(np.nan)
        else:
            vertices = np.asarray(vertices, np.float_)

        if codes is not None:
            codes = np.asarray(codes, self.code_type)
            assert codes.ndim == 1
            assert len(codes) == len(vertices)
            if len(codes):
                assert codes[0] == self.MOVETO
        elif closed:
            codes = np.empty(len(vertices), dtype=self.code_type)
            codes[0] = self.MOVETO
            codes[1:-1] = self.LINETO
            codes[-1] = self.CLOSEPOLY

        assert vertices.ndim == 2
        assert vertices.shape[1] == 2

        self.should_simplify = (rcParams['path.simplify'] and
                                (len(vertices) >= 128 and
                                 (codes is None or np.all(codes <= Path.LINETO))))
        self.simplify_threshold = rcParams['path.simplify_threshold']
        self.has_nonfinite = not np.isfinite(vertices).all()
        self.codes = codes
        self.vertices = vertices
        self._interpolation_steps = _interpolation_steps

    @classmethod
    def make_compound_path_from_polys(cls, XY):
        """
        (static method) Make a compound path object to draw a number
        of polygons with equal numbers of sides XY is a (numpolys x
        numsides x 2) numpy array of vertices.  Return object is a
        :class:`Path`

        .. plot:: mpl_examples/api/histogram_path_demo.py

        """

        # for each poly: 1 for the MOVETO, (numsides-1) for the LINETO, 1 for the
        # CLOSEPOLY; the vert for the closepoly is ignored but we still need
        # it to keep the codes aligned with the vertices
        numpolys, numsides, two = XY.shape
        assert(two==2)
        stride = numsides + 1
        nverts = numpolys * stride
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * cls.LINETO
        codes[0::stride] = cls.MOVETO
        codes[numsides::stride] = cls.CLOSEPOLY
        for i in range(numsides):
            verts[i::stride] = XY[:,i]

        return cls(verts, codes)

    @classmethod
    def make_compound_path(cls, *args):
        """
        (staticmethod) Make a compound path from a list of Path
        objects.  Only polygons (not curves) are supported.
        """
        for p in args:
            assert p.codes is None

        lengths = [len(x) for x in args]
        total_length = sum(lengths)

        vertices = np.vstack([x.vertices for x in args])
        vertices.reshape((total_length, 2))

        codes = cls.LINETO * np.ones(total_length)
        i = 0
        for length in lengths:
            codes[i] = cls.MOVETO
            i += length

        return cls(vertices, codes)

    def __repr__(self):
        return "Path(%s, %s)" % (self.vertices, self.codes)

    def __len__(self):
        return len(self.vertices)

    def iter_segments(self, transform=None, remove_nans=True, clip=None,
                      snap=False, stroke_width=1.0, simplify=None,
                      curves=True):
        """
        Iterates over all of the curve segments in the path.  Each
        iteration returns a 2-tuple (*vertices*, *code*), where
        *vertices* is a sequence of 1 - 3 coordinate pairs, and *code* is
        one of the :class:`Path` codes.

        Additionally, this method can provide a number of standard
        cleanups and conversions to the path.

        *transform*: if not None, the given affine transformation will
         be applied to the path.

        *remove_nans*: if True, will remove all NaNs from the path and
         insert MOVETO commands to skip over them.

        *clip*: if not None, must be a four-tuple (x1, y1, x2, y2)
         defining a rectangle in which to clip the path.

        *snap*: if None, auto-snap to pixels, to reduce
         fuzziness of rectilinear lines.  If True, force snapping, and
         if False, don't snap.

        *stroke_width*: the width of the stroke being drawn.  Needed
         as a hint for the snapping algorithm.

        *simplify*: if True, perform simplification, to remove
         vertices that do not affect the appearance of the path.  If
         False, perform no simplification.  If None, use the
         should_simplify member variable.

        *curves*: If True, curve segments will be returned as curve
         segments.  If False, all curves will be converted to line
         segments.
        """
        vertices = self.vertices
        if not len(vertices):
            return

        codes        = self.codes

        NUM_VERTICES = self.NUM_VERTICES
        MOVETO       = self.MOVETO
        LINETO       = self.LINETO
        CLOSEPOLY    = self.CLOSEPOLY
        STOP         = self.STOP

        vertices, codes = cleanup_path(self, transform, remove_nans, clip,
                                       snap, stroke_width, simplify, curves)
        len_vertices = len(vertices)

        i = 0
        while i < len_vertices:
            code = codes[i]
            if code == STOP:
                return
            else:
                num_vertices = NUM_VERTICES[int(code) & 0xf]
                curr_vertices = vertices[i:i+num_vertices].flatten()
                yield curr_vertices, code
                i += num_vertices

    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        .. seealso::

            :class:`matplotlib.transforms.TransformedPath`
                A specialized path class that will cache the
                transformed result and automatically update when the
                transform changes.
        """
        return Path(transform.transform(self.vertices), self.codes,
                    self._interpolation_steps)

    def contains_point(self, point, transform=None, radius=0.0):
        """
        Returns *True* if the path contains the given point.

        If *transform* is not *None*, the path will be transformed
        before performing the test.
        """
        if transform is not None:
            transform = transform.frozen()
        result = point_in_path(point[0], point[1], radius, self, transform)
        return result

    def contains_path(self, path, transform=None):
        """
        Returns *True* if this path completely contains the given path.

        If *transform* is not *None*, the path will be transformed
        before performing the test.
        """
        if transform is not None:
            transform = transform.frozen()
        return path_in_path(self, None, path, transform)

    def get_extents(self, transform=None):
        """
        Returns the extents (*xmin*, *ymin*, *xmax*, *ymax*) of the
        path.

        Unlike computing the extents on the *vertices* alone, this
        algorithm will take into account the curves and deal with
        control points appropriately.
        """
        from transforms import Bbox
        path = self
        if transform is not None:
            transform = transform.frozen()
            if not transform.is_affine:
                path = self.transformed(transform)
                transform = None
        return Bbox(get_path_extents(path, transform))

    def intersects_path(self, other, filled=True):
        """
        Returns *True* if this path intersects another given path.

        *filled*, when True, treats the paths as if they were filled.
        That is, if one path completely encloses the other,
        :meth:`intersects_path` will return True.
        """
        return path_intersects_path(self, other, filled)

    def intersects_bbox(self, bbox, filled=True):
        """
        Returns *True* if this path intersects a given
        :class:`~matplotlib.transforms.Bbox`.

        *filled*, when True, treats the path as if it was filled.
        That is, if one path completely encloses the other,
        :meth:`intersects_path` will return True.
        """
        from transforms import BboxTransformTo
        rectangle = self.unit_rectangle().transformed(
            BboxTransformTo(bbox))
        result = self.intersects_path(rectangle, filled)
        return result

    def interpolated(self, steps):
        """
        Returns a new path resampled to length N x steps.  Does not
        currently handle interpolating curves.
        """
        if steps == 1:
            return self

        vertices = simple_linear_interpolation(self.vertices, steps)
        codes = self.codes
        if codes is not None:
            new_codes = Path.LINETO * np.ones(((len(codes) - 1) * steps + 1, ))
            new_codes[0::steps] = codes
        else:
            new_codes = None
        return Path(vertices, new_codes)

    def to_polygons(self, transform=None, width=0, height=0):
        """
        Convert this path to a list of polygons.  Each polygon is an
        Nx2 array of vertices.  In other words, each polygon has no
        ``MOVETO`` instructions or curves.  This is useful for
        displaying in backends that do not support compound paths or
        Bezier curves, such as GDK.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.
        """
        if len(self.vertices) == 0:
            return []

        if transform is not None:
            transform = transform.frozen()

        if self.codes is None and (width == 0 or height == 0):
            if transform is None:
                return [self.vertices]
            else:
                return [transform.transform(self.vertices)]

        # Deal with the case where there are curves and/or multiple
        # subpaths (using extension code)
        return convert_path_to_polygons(self, transform, width, height)

    _unit_rectangle = None
    @classmethod
    def unit_rectangle(cls):
        """
        (staticmethod) Returns a :class:`Path` of the unit rectangle
        from (0, 0) to (1, 1).
        """
        if cls._unit_rectangle is None:
            cls._unit_rectangle = \
                cls([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                    [cls.MOVETO, cls.LINETO, cls.LINETO, cls.LINETO, cls.CLOSEPOLY])
        return cls._unit_rectangle

    _unit_regular_polygons = WeakValueDictionary()

    @classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        (staticmethod) Returns a :class:`Path` for a unit regular
        polygon with the given *numVertices* and radius of 1.0,
        centered at (0, 0).
        """
        if numVertices <= 16:
            path = cls._unit_regular_polygons.get(numVertices)
        else:
            path = None
        if path is None:
            theta = (2*np.pi/numVertices *
                     np.arange(numVertices + 1).reshape((numVertices + 1, 1)))
            # This initial rotation is to make sure the polygon always
            # "points-up"
            theta += np.pi / 2.0
            verts = np.concatenate((np.cos(theta), np.sin(theta)), 1)
            codes = np.empty((numVertices + 1,))
            codes[0] = cls.MOVETO
            codes[1:-1] = cls.LINETO
            codes[-1] = cls.CLOSEPOLY
            path = cls(verts, codes)
            if numVertices <= 16:
                cls._unit_regular_polygons[numVertices] = path
        return path

    _unit_regular_stars = WeakValueDictionary()

    @classmethod
    def unit_regular_star(cls, numVertices, innerCircle=0.5):
        """
        (staticmethod) Returns a :class:`Path` for a unit regular star
        with the given numVertices and radius of 1.0, centered at (0,
        0).
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
            verts = np.vstack((r*np.cos(theta), r*np.sin(theta))).transpose()
            codes = np.empty((ns2 + 1,))
            codes[0] = cls.MOVETO
            codes[1:-1] = cls.LINETO
            codes[-1] = cls.CLOSEPOLY
            path = cls(verts, codes)
            if numVertices <= 16:
                cls._unit_regular_polygons[(numVertices, innerCircle)] = path
        return path

    @classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        (staticmethod) Returns a :class:`Path` for a unit regular
        asterisk with the given numVertices and radius of 1.0,
        centered at (0, 0).
        """
        return cls.unit_regular_star(numVertices, 0.0)

    _unit_circle = None

    @classmethod
    def unit_circle(cls):
        """
        (staticmethod) Returns a :class:`Path` of the unit circle.
        The circle is approximated using cubic Bezier curves.  This
        uses 8 splines around the circle using the approach presented
        here:

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.
        """
        if cls._unit_circle is None:
            MAGIC = 0.2652031
            SQRTHALF = np.sqrt(0.5)
            MAGIC45 = np.sqrt((MAGIC*MAGIC) / 2.0)

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
                np.float_)

            codes = cls.CURVE4 * np.ones(26)
            codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

            cls._unit_circle = cls(vertices, codes)
        return cls._unit_circle

    _unit_circle_righthalf = None

    @classmethod
    def unit_circle_righthalf(cls):
        """
        (staticmethod) Returns a :class:`Path` of the right half
        of a unit circle. The circle is approximated using cubic Bezier
        curves.  This uses 4 splines around the circle using the approach
        presented here:

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.
        """
        if cls._unit_circle_righthalf is None:
            MAGIC = 0.2652031
            SQRTHALF = np.sqrt(0.5)
            MAGIC45 = np.sqrt((MAGIC*MAGIC) / 2.0)

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

                np.float_)

            codes = cls.CURVE4 * np.ones(14)
            codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

            cls._unit_circle_righthalf = cls(vertices, codes)
        return cls._unit_circle_righthalf

    @classmethod
    def arc(cls, theta1, theta2, n=None, is_wedge=False):
        """
        (staticmethod) Returns an arc on the unit circle from angle
        *theta1* to angle *theta2* (in degrees).

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

           Masionobe, L.  2003.  `Drawing an elliptical arc using
           polylines, quadratic or cubic Bezier curves
           <http://www.spaceroots.org/documents/ellipse/index.html>`_.
        """
        # degrees to radians
        theta1 *= np.pi / 180.0
        theta2 *= np.pi / 180.0

        twopi  = np.pi * 2.0
        halfpi = np.pi * 0.5

        eta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
        eta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
        eta2 -= twopi * np.floor((eta2 - eta1) / twopi)
        if (theta2 - theta1 > np.pi) and (eta2 - eta1 < np.pi):
            eta2 += twopi

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
            vertices = np.zeros((length, 2), np.float_)
            codes = cls.CURVE4 * np.ones((length, ), cls.code_type)
            vertices[1] = [xA[0], yA[0]]
            codes[0:2] = [cls.MOVETO, cls.LINETO]
            codes[-2:] = [cls.LINETO, cls.CLOSEPOLY]
            vertex_offset = 2
            end = length - 2
        else:
            length = n * 3 + 1
            vertices = np.empty((length, 2), np.float_)
            codes = cls.CURVE4 * np.ones((length, ), cls.code_type)
            vertices[0] = [xA[0], yA[0]]
            codes[0] = cls.MOVETO
            vertex_offset = 1
            end = length

        vertices[vertex_offset  :end:3, 0] = xA + alpha * xA_dot
        vertices[vertex_offset  :end:3, 1] = yA + alpha * yA_dot
        vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
        vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
        vertices[vertex_offset+2:end:3, 0] = xB
        vertices[vertex_offset+2:end:3, 1] = yB

        return cls(vertices, codes)

    @classmethod
    def wedge(cls, theta1, theta2, n=None):
        """
        (staticmethod) Returns a wedge of the unit circle from angle
        *theta1* to angle *theta2* (in degrees).

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.
        """
        return cls.arc(theta1, theta2, n, True)

    _hatch_dict = maxdict(8)

    @classmethod
    def hatch(cls, hatchpattern, density=6):
        """
        Given a hatch specifier, *hatchpattern*, generates a Path that
        can be used in a repeated hatching pattern.  *density* is the
        number of lines per unit square.
        """
        from matplotlib.hatch import get_path

        if hatchpattern is None:
            return None

        hatch_path = cls._hatch_dict.get((hatchpattern, density))
        if hatch_path is not None:
            return hatch_path

        hatch_path = get_path(hatchpattern, density)
        cls._hatch_dict[(hatchpattern, density)] = hatch_path
        return hatch_path

_get_path_collection_extents = get_path_collection_extents
def get_path_collection_extents(*args):
    """
    Given a sequence of :class:`Path` objects, returns the bounding
    box that encapsulates all of them.
    """
    from transforms import Bbox
    if len(args[1]) == 0:
        raise ValueError("No paths provided")
    return Bbox.from_extents(*_get_path_collection_extents(*args))
