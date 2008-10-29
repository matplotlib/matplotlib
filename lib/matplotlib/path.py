"""
Contains a class for managing paths (polylines).
"""

import math
from weakref import WeakValueDictionary

import numpy as np
from numpy import ma

from matplotlib._path import point_in_path, get_path_extents, \
    point_in_path_collection, get_path_collection_extents, \
    path_in_path, path_intersects_path, convert_path_to_polygons
from matplotlib.cbook import simple_linear_interpolation

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

    Note also that the vertices and codes arrays should be treated as
    immutable -- there are a number of optimizations and assumptions
    made up front in the constructor that will not change when the
    data changes.
    """

    # Path codes
    STOP      = 0 # 1 vertex
    MOVETO    = 1 # 1 vertex
    LINETO    = 2 # 1 vertex
    CURVE3    = 3 # 2 vertices
    CURVE4    = 4 # 3 vertices
    CLOSEPOLY = 5 # 1 vertex

    NUM_VERTICES = [1, 1, 1, 2, 3, 1]

    code_type = np.uint8

    def __init__(self, vertices, codes=None):
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
        """
        if ma.isMaskedArray(vertices):
            vertices = vertices.astype(np.float_).filled(np.nan)
        else:
            vertices = np.asarray(vertices, np.float_)

        if codes is not None:
            codes = np.asarray(codes, self.code_type)
            assert codes.ndim == 1
            assert len(codes) == len(vertices)

        assert vertices.ndim == 2
        assert vertices.shape[1] == 2

        self.should_simplify = (len(vertices) >= 128 and
                                (codes is None or np.all(codes <= Path.LINETO)))
        self.has_nonfinite = not np.isfinite(vertices).all()
        self.codes = codes
        self.vertices = vertices

    #@staticmethod
    def make_compound_path(*args):
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

        codes = Path.LINETO * np.ones(total_length)
        i = 0
        for length in lengths:
            codes[i] = Path.MOVETO
            i += length

        return Path(vertices, codes)
    make_compound_path = staticmethod(make_compound_path)

    def __repr__(self):
        return "Path(%s, %s)" % (self.vertices, self.codes)

    def __len__(self):
        return len(self.vertices)

    def iter_segments(self, simplify=None):
        """
        Iterates over all of the curve segments in the path.  Each
        iteration returns a 2-tuple (*vertices*, *code*), where
        *vertices* is a sequence of 1 - 3 coordinate pairs, and *code* is
        one of the :class:`Path` codes.

        If *simplify* is provided, it must be a tuple (*width*,
        *height*) defining the size of the figure, in native units
        (e.g. pixels or points).  Simplification implies both removing
        adjacent line segments that are very close to parallel, and
        removing line segments outside of the figure.  The path will
        be simplified *only* if :attr:`should_simplify` is True, which
        is determined in the constructor by this criteria:

           - No curves
           - More than 128 vertices
        """
        vertices = self.vertices
        if not len(vertices):
            return

        codes        = self.codes
        len_vertices = len(vertices)
        isfinite     = np.isfinite

        NUM_VERTICES = self.NUM_VERTICES
        MOVETO       = self.MOVETO
        LINETO       = self.LINETO
        CLOSEPOLY    = self.CLOSEPOLY
        STOP         = self.STOP

        if simplify is not None and self.should_simplify:
            polygons = self.to_polygons(None, *simplify)
            for vertices in polygons:
                yield vertices[0], MOVETO
                for v in vertices[1:]:
                    yield v, LINETO
        elif codes is None:
            if self.has_nonfinite:
                next_code = MOVETO
                for v in vertices:
                    if np.isfinite(v).all():
                        yield v, next_code
                        next_code = LINETO
                    else:
                        next_code = MOVETO
            else:
                yield vertices[0], MOVETO
                for v in vertices[1:]:
                    yield v, LINETO
        else:
            i = 0
            was_nan = False
            while i < len_vertices:
                code = codes[i]
                if code == CLOSEPOLY:
                    yield [], code
                    i += 1
                elif code == STOP:
                    return
                else:
                    num_vertices = NUM_VERTICES[int(code)]
                    curr_vertices = vertices[i:i+num_vertices].flatten()
                    if not isfinite(curr_vertices).all():
                        was_nan = True
                    elif was_nan:
                        yield curr_vertices[-2:], MOVETO
                        was_nan = False
                    else:
                        yield curr_vertices, code
                    i += num_vertices

    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        .. seealso::
            :class:`matplotlib.transforms.TransformedPath`:
                A specialized path class that will cache the
                transformed result and automatically update when the
                transform changes.
        """
        return Path(transform.transform(self.vertices), self.codes)

    def contains_point(self, point, transform=None):
        """
        Returns *True* if the path contains the given point.

        If *transform* is not *None*, the path will be transformed
        before performing the test.
        """
        if transform is not None:
            transform = transform.frozen()
        return point_in_path(point[0], point[1], self, transform)

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
        if transform is not None:
            transform = transform.frozen()
        return Bbox(get_path_extents(self, transform))

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
    #@classmethod
    def unit_rectangle(cls):
        """
        (staticmethod) Returns a :class:`Path` of the unit rectangle
        from (0, 0) to (1, 1).
        """
        if cls._unit_rectangle is None:
            cls._unit_rectangle = \
                Path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        return cls._unit_rectangle
    unit_rectangle = classmethod(unit_rectangle)

    _unit_regular_polygons = WeakValueDictionary()
    #@classmethod
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
            path = Path(verts)
            cls._unit_regular_polygons[numVertices] = path
        return path
    unit_regular_polygon = classmethod(unit_regular_polygon)

    _unit_regular_stars = WeakValueDictionary()
    #@classmethod
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
            path = Path(verts)
            cls._unit_regular_polygons[(numVertices, innerCircle)] = path
        return path
    unit_regular_star = classmethod(unit_regular_star)

    #@classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        (staticmethod) Returns a :class:`Path` for a unit regular
        asterisk with the given numVertices and radius of 1.0,
        centered at (0, 0).
        """
        return cls.unit_regular_star(numVertices, 0.0)
    unit_regular_asterisk = classmethod(unit_regular_asterisk)

    _unit_circle = None
    #@classmethod
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

            cls._unit_circle = Path(vertices, codes)
        return cls._unit_circle
    unit_circle = classmethod(unit_circle)

    #@classmethod
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
            codes = Path.CURVE4 * np.ones((length, ), Path.code_type)
            vertices[1] = [xA[0], yA[0]]
            codes[0:2] = [Path.MOVETO, Path.LINETO]
            codes[-2:] = [Path.LINETO, Path.CLOSEPOLY]
            vertex_offset = 2
            end = length - 2
        else:
            length = n * 3 + 1
            vertices = np.zeros((length, 2), np.float_)
            codes = Path.CURVE4 * np.ones((length, ), Path.code_type)
            vertices[0] = [xA[0], yA[0]]
            codes[0] = Path.MOVETO
            vertex_offset = 1
            end = length

        vertices[vertex_offset  :end:3, 0] = xA + alpha * xA_dot
        vertices[vertex_offset  :end:3, 1] = yA + alpha * yA_dot
        vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
        vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
        vertices[vertex_offset+2:end:3, 0] = xB
        vertices[vertex_offset+2:end:3, 1] = yB

        return Path(vertices, codes)
    arc = classmethod(arc)

    #@classmethod
    def wedge(cls, theta1, theta2, n=None):
        """
        (staticmethod) Returns a wedge of the unit circle from angle
        *theta1* to angle *theta2* (in degrees).

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.
        """
        return cls.arc(theta1, theta2, n, True)
    wedge = classmethod(wedge)

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
