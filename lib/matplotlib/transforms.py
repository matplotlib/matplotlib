"""
matplotlib includes a framework for arbitrary geometric
transformations that is used determine the final position of all
elements drawn on the canvas.

Transforms are composed into trees of :class:`TransformNode` objects
whose actual value depends on their children.  When the contents of
children change, their parents are automatically invalidated.  The
next time an invalidated transform is accessed, it is recomputed to
reflect those changes.  This invalidation/caching approach prevents
unnecessary recomputations of transforms, and contributes to better
interactive performance.

For example, here is a graph of the transform tree used to plot data
to the graph:

.. image:: ../_static/transforms.png

The framework can be used for both affine and non-affine
transformations.  However, for speed, we want use the backend
renderers to perform affine transformations whenever possible.
Therefore, it is possible to perform just the affine or non-affine
part of a transformation on a set of data.  The affine is always
assumed to occur after the non-affine.  For any transform::

  full transform == non-affine part + affine part

The backends are not expected to handle non-affine transformations
themselves.
"""

from __future__ import print_function, division
import numpy as np
from numpy import ma
from matplotlib._path import (affine_transform, count_bboxes_overlapping_bbox,
    update_path_extents)
from numpy.linalg import inv

from weakref import WeakValueDictionary
import warnings
try:
    set
except NameError:
    from sets import Set as set

from path import Path

DEBUG = False

MaskedArray = ma.MaskedArray


class TransformNode(object):
    """
    :class:`TransformNode` is the base class for anything that
    participates in the transform tree and needs to invalidate its
    parents or be invalidated.  This includes classes that are not
    really transforms, such as bounding boxes, since some transforms
    depend on bounding boxes to compute their values.
    """
    _gid = 0

    # Invalidation may affect only the affine part.  If the
    # invalidation was "affine-only", the _invalid member is set to
    # INVALID_AFFINE_ONLY
    INVALID_NON_AFFINE = 1
    INVALID_AFFINE = 2
    INVALID = INVALID_NON_AFFINE | INVALID_AFFINE

    # Some metadata about the transform, used to determine whether an
    # invalidation is affine-only
    is_affine = False
    is_bbox = False

    pass_through = False
    """
    If pass_through is True, all ancestors will always be
    invalidated, even if 'self' is already invalid.
    """

    def __init__(self, shorthand_name=None):
        """
        Creates a new :class:`TransformNode`.

        **shorthand_name** - a string representing the "name" of this
                             transform. The name carries no significance
                             other than to improve the readability of
                             ``str(transform)`` when DEBUG=True.
        """
        # Parents are stored in a WeakValueDictionary, so that if the
        # parents are deleted, references from the children won't keep
        # them alive.
        self._parents = WeakValueDictionary()

        # TransformNodes start out as invalid until their values are
        # computed for the first time.
        self._invalid = 1
        self._shorthand_name = shorthand_name or ''

    if DEBUG:
        def __str__(self):
            # either just return the name of this TransformNode, or it's repr
            return self._shorthand_name or repr(self)

    def __getstate__(self):
        d = self.__dict__.copy()
        # turn the weakkey dictionary into a normal dictionary
        d['_parents'] = dict(self._parents.iteritems())
        return d

    def __setstate__(self, data_dict):
        self.__dict__ = data_dict
        # turn the normal dictionary back into a WeakValueDictionary
        self._parents = WeakValueDictionary(self._parents)

    def __copy__(self, *args):
        raise NotImplementedError(
            "TransformNode instances can not be copied. " +
            "Consider using frozen() instead.")
    __deepcopy__ = __copy__

    def invalidate(self):
        """
        Invalidate this :class:`TransformNode` and triggers an
        invalidation of its ancestors.  Should be called any
        time the transform changes.
        """
        value = self.INVALID
        if self.is_affine:
            value = self.INVALID_AFFINE
        return self._invalidate_internal(value, invalidating_node=self)

    def _invalidate_internal(self, value, invalidating_node):
        """
        Called by :meth:`invalidate` and subsequently ascends the transform
        stack calling each TransformNode's _invalidate_internal method.
        """
        # determine if this call will be an extension to the invalidation
        # status. If not, then a shortcut means that we needn't invoke an
        # invalidation up the transform stack as it will already have been
        # invalidated.

        # N.B This makes the invalidation sticky, once a transform has been
        # invalidated as NON_AFFINE, then it will always be invalidated as
        # NON_AFFINE even when triggered with a AFFINE_ONLY invalidation.
        # In most cases this is not a problem (i.e. for interactive panning and
        # zooming) and the only side effect will be on performance.
        status_changed = self._invalid < value

        if self.pass_through or status_changed:
            self._invalid = value

            for parent in self._parents.values():
                parent._invalidate_internal(value=value,
                                            invalidating_node=self)

    def set_children(self, *children):
        """
        Set the children of the transform, to let the invalidation
        system know which transforms can invalidate this transform.
        Should be called from the constructor of any transforms that
        depend on other transforms.
        """
        for child in children:
            child._parents[id(self)] = self

    if DEBUG:
        _set_children = set_children

        def set_children(self, *children):
            self._set_children(*children)
            self._children = children
        set_children.__doc__ = _set_children.__doc__

    def frozen(self):
        """
        Returns a frozen copy of this transform node.  The frozen copy
        will not update when its children change.  Useful for storing
        a previously known state of a transform where
        ``copy.deepcopy()`` might normally be used.
        """
        return self

    if DEBUG:
        def write_graphviz(self, fobj, highlight=[]):
            """
            For debugging purposes.

            Writes the transform tree rooted at 'self' to a graphviz "dot"
            format file.  This file can be run through the "dot" utility
            to produce a graph of the transform tree.

            Affine transforms are marked in blue.  Bounding boxes are
            marked in yellow.

            *fobj*: A Python file-like object

            Once the "dot" file has been created, it can be turned into a
            png easily with::

                $> dot -Tpng -o $OUTPUT_FILE $DOT_FILE

            """
            seen = set()

            def recurse(root):
                if root in seen:
                    return
                seen.add(root)
                props = {}
                label = root.__class__.__name__
                if root._invalid:
                    label = '[%s]' % label
                if root in highlight:
                    props['style'] = 'bold'
                props['shape'] = 'box'
                props['label'] = '"%s"' % label
                props = ' '.join(['%s=%s' % (key, val)
                                  for key, val
                                  in props.iteritems()])

                fobj.write('%s [%s];\n' %
                           (hash(root), props))

                if hasattr(root, '_children'):
                    for child in root._children:
                        name = '?'
                        for key, val in root.__dict__.iteritems():
                            if val is child:
                                name = key
                                break
                        fobj.write('"%s" -> "%s" [label="%s", fontsize=10];\n'
                                    % (hash(root),
                                    hash(child),
                                    name))
                        recurse(child)

            fobj.write("digraph G {\n")
            recurse(self)
            fobj.write("}\n")


class BboxBase(TransformNode):
    """
    This is the base class of all bounding boxes, and provides
    read-only access to its data.  A mutable bounding box is provided
    by the :class:`Bbox` class.

    The canonical representation is as two points, with no
    restrictions on their ordering.  Convenience properties are
    provided to get the left, bottom, right and top edges and width
    and height, but these are not stored explicitly.
    """
    is_bbox = True
    is_affine = True

    #* Redundant: Removed for performance
    #
    # def __init__(self):
    #     TransformNode.__init__(self)

    if DEBUG:
        def _check(points):
            if ma.isMaskedArray(points):
                warnings.warn("Bbox bounds are a masked array.")
            points = np.asarray(points)
            if (points[1, 0] - points[0, 0] == 0 or
                points[1, 1] - points[0, 1] == 0):
                warnings.warn("Singular Bbox.")
        _check = staticmethod(_check)

    def frozen(self):
        return Bbox(self.get_points().copy())
    frozen.__doc__ = TransformNode.__doc__

    def __array__(self, *args, **kwargs):
        return self.get_points()

    def is_unit(self):
        """
        Returns True if the :class:`Bbox` is the unit bounding box
        from (0, 0) to (1, 1).
        """
        return list(self.get_points().flatten()) == [0., 0., 1., 1.]

    def _get_x0(self):
        return self.get_points()[0, 0]
    x0 = property(_get_x0, None, None, """
         (property) :attr:`x0` is the first of the pair of *x* coordinates that
         define the bounding box.  :attr:`x0` is not guaranteed to be
         less than :attr:`x1`.  If you require that, use :attr:`xmin`.""")

    def _get_y0(self):
        return self.get_points()[0, 1]
    y0 = property(_get_y0, None, None, """
         (property) :attr:`y0` is the first of the pair of *y* coordinates that
         define the bounding box.  :attr:`y0` is not guaranteed to be
         less than :attr:`y1`.  If you require that, use :attr:`ymin`.""")

    def _get_x1(self):
        return self.get_points()[1, 0]
    x1 = property(_get_x1, None, None, """
         (property) :attr:`x1` is the second of the pair of *x* coordinates
         that define the bounding box.  :attr:`x1` is not guaranteed to be
         greater than :attr:`x0`.  If you require that, use :attr:`xmax`.""")

    def _get_y1(self):
        return self.get_points()[1, 1]
    y1 = property(_get_y1, None, None, """
         (property) :attr:`y1` is the second of the pair of *y* coordinates
         that define the bounding box.  :attr:`y1` is not guaranteed to be
         greater than :attr:`y0`.  If you require that, use :attr:`ymax`.""")

    def _get_p0(self):
        return self.get_points()[0]
    p0 = property(_get_p0, None, None, """
         (property) :attr:`p0` is the first pair of (*x*, *y*) coordinates
         that define the bounding box.  It is not guaranteed to be the
         bottom-left corner.  For that, use :attr:`min`.""")

    def _get_p1(self):
        return self.get_points()[1]
    p1 = property(_get_p1, None, None, """
         (property) :attr:`p1` is the second pair of (*x*, *y*) coordinates
         that define the bounding box.  It is not guaranteed to be the
         top-right corner.  For that, use :attr:`max`.""")

    def _get_xmin(self):
        return min(self.get_points()[:, 0])
    xmin = property(_get_xmin, None, None, """
        (property) :attr:`xmin` is the left edge of the bounding box.""")

    def _get_ymin(self):
        return min(self.get_points()[:, 1])
    ymin = property(_get_ymin, None, None, """
        (property) :attr:`ymin` is the bottom edge of the bounding box.""")

    def _get_xmax(self):
        return max(self.get_points()[:, 0])
    xmax = property(_get_xmax, None, None, """
        (property) :attr:`xmax` is the right edge of the bounding box.""")

    def _get_ymax(self):
        return max(self.get_points()[:, 1])
    ymax = property(_get_ymax, None, None, """
        (property) :attr:`ymax` is the top edge of the bounding box.""")

    def _get_min(self):
        return [min(self.get_points()[:, 0]),
                min(self.get_points()[:, 1])]
    min = property(_get_min, None, None, """
        (property) :attr:`min` is the bottom-left corner of the bounding
        box.""")

    def _get_max(self):
        return [max(self.get_points()[:, 0]),
                max(self.get_points()[:, 1])]
    max = property(_get_max, None, None, """
        (property) :attr:`max` is the top-right corner of the bounding box.""")

    def _get_intervalx(self):
        return self.get_points()[:, 0]
    intervalx = property(_get_intervalx, None, None, """
        (property) :attr:`intervalx` is the pair of *x* coordinates that define
        the bounding box. It is not guaranteed to be sorted from left to
        right.""")

    def _get_intervaly(self):
        return self.get_points()[:, 1]
    intervaly = property(_get_intervaly, None, None, """
        (property) :attr:`intervaly` is the pair of *y* coordinates that define
        the bounding box.  It is not guaranteed to be sorted from bottom to
        top.""")

    def _get_width(self):
        points = self.get_points()
        return points[1, 0] - points[0, 0]
    width = property(_get_width, None, None, """
        (property) The width of the bounding box.  It may be negative if
        :attr:`x1` < :attr:`x0`.""")

    def _get_height(self):
        points = self.get_points()
        return points[1, 1] - points[0, 1]
    height = property(_get_height, None, None, """
        (property) The height of the bounding box.  It may be negative if
        :attr:`y1` < :attr:`y0`.""")

    def _get_size(self):
        points = self.get_points()
        return points[1] - points[0]
    size = property(_get_size, None, None, """
        (property) The width and height of the bounding box.  May be negative,
        in the same way as :attr:`width` and :attr:`height`.""")

    def _get_bounds(self):
        x0, y0, x1, y1 = self.get_points().flatten()
        return (x0, y0, x1 - x0, y1 - y0)
    bounds = property(_get_bounds, None, None, """
        (property) Returns (:attr:`x0`, :attr:`y0`, :attr:`width`,
        :attr:`height`).""")

    def _get_extents(self):
        return self.get_points().flatten().copy()
    extents = property(_get_extents, None, None, """
        (property) Returns (:attr:`x0`, :attr:`y0`, :attr:`x1`,
        :attr:`y1`).""")

    def get_points(self):
        return NotImplementedError()

    def containsx(self, x):
        """
        Returns True if *x* is between or equal to :attr:`x0` and
        :attr:`x1`.
        """
        x0, x1 = self.intervalx
        return ((x0 < x1
                 and (x >= x0 and x <= x1))
                or (x >= x1 and x <= x0))

    def containsy(self, y):
        """
        Returns True if *y* is between or equal to :attr:`y0` and
        :attr:`y1`.
        """
        y0, y1 = self.intervaly
        return ((y0 < y1
                 and (y >= y0 and y <= y1))
                or (y >= y1 and y <= y0))

    def contains(self, x, y):
        """
        Returns *True* if (*x*, *y*) is a coordinate inside the
        bounding box or on its edge.
        """
        return self.containsx(x) and self.containsy(y)

    def overlaps(self, other):
        """
        Returns True if this bounding box overlaps with the given
        bounding box *other*.
        """
        ax1, ay1, ax2, ay2 = self._get_extents()
        bx1, by1, bx2, by2 = other._get_extents()

        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2

        return not ((bx2 < ax1) or
                    (by2 < ay1) or
                    (bx1 > ax2) or
                    (by1 > ay2))

    def fully_containsx(self, x):
        """
        Returns True if *x* is between but not equal to :attr:`x0` and
        :attr:`x1`.
        """
        x0, x1 = self.intervalx
        return ((x0 < x1
                 and (x > x0 and x < x1))
                or (x > x1 and x < x0))

    def fully_containsy(self, y):
        """
        Returns True if *y* is between but not equal to :attr:`y0` and
        :attr:`y1`.
        """
        y0, y1 = self.intervaly
        return ((y0 < y1
                 and (y > y0 and y < y1))
                or (y > y1 and y < y0))

    def fully_contains(self, x, y):
        """
        Returns True if (*x*, *y*) is a coordinate inside the bounding
        box, but not on its edge.
        """
        return self.fully_containsx(x) \
            and self.fully_containsy(y)

    def fully_overlaps(self, other):
        """
        Returns True if this bounding box overlaps with the given
        bounding box *other*, but not on its edge alone.
        """
        ax1, ay1, ax2, ay2 = self._get_extents()
        bx1, by1, bx2, by2 = other._get_extents()

        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2

        return not ((bx2 <= ax1) or
                    (by2 <= ay1) or
                    (bx1 >= ax2) or
                    (by1 >= ay2))

    def transformed(self, transform):
        """
        Return a new :class:`Bbox` object, statically transformed by
        the given transform.
        """
        return Bbox(transform.transform(self.get_points()))

    def inverse_transformed(self, transform):
        """
        Return a new :class:`Bbox` object, statically transformed by
        the inverse of the given transform.
        """
        return Bbox(transform.inverted().transform(self.get_points()))

    coefs = {'C':  (0.5, 0.5),
             'SW': (0, 0),
             'S':  (0.5, 0),
             'SE': (1.0, 0),
             'E':  (1.0, 0.5),
             'NE': (1.0, 1.0),
             'N':  (0.5, 1.0),
             'NW': (0, 1.0),
             'W':  (0, 0.5)}

    def anchored(self, c, container=None):
        """
        Return a copy of the :class:`Bbox`, shifted to position *c*
        within a container.

        *c*: may be either:

          * a sequence (*cx*, *cy*) where *cx* and *cy* range from 0
            to 1, where 0 is left or bottom and 1 is right or top

          * a string:
            - 'C' for centered
            - 'S' for bottom-center
            - 'SE' for bottom-left
            - 'E' for left
            - etc.

        Optional argument *container* is the box within which the
        :class:`Bbox` is positioned; it defaults to the initial
        :class:`Bbox`.
        """
        if container is None:
            container = self
        l, b, w, h = container.bounds
        if isinstance(c, basestring):
            cx, cy = self.coefs[c]
        else:
            cx, cy = c
        L, B, W, H = self.bounds
        return Bbox(self._points +
                    [(l + cx * (w - W)) - L,
                     (b + cy * (h - H)) - B])

    def shrunk(self, mx, my):
        """
        Return a copy of the :class:`Bbox`, shrunk by the factor *mx*
        in the *x* direction and the factor *my* in the *y* direction.
        The lower left corner of the box remains unchanged.  Normally
        *mx* and *my* will be less than 1, but this is not enforced.
        """
        w, h = self.size
        return Bbox([self._points[0],
                    self._points[0] + [mx * w, my * h]])

    def shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
        """
        Return a copy of the :class:`Bbox`, shrunk so that it is as
        large as it can be while having the desired aspect ratio,
        *box_aspect*.  If the box coordinates are relative---that
        is, fractions of a larger box such as a figure---then the
        physical aspect ratio of that figure is specified with
        *fig_aspect*, so that *box_aspect* can also be given as a
        ratio of the absolute dimensions, not the relative dimensions.
        """
        assert box_aspect > 0 and fig_aspect > 0
        if container is None:
            container = self
        w, h = container.size
        H = w * box_aspect / fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect / box_aspect
            H = h
        return Bbox([self._points[0],
                     self._points[0] + (W, H)])

    def splitx(self, *args):
        """
        e.g., ``bbox.splitx(f1, f2, ...)``

        Returns a list of new :class:`Bbox` objects formed by
        splitting the original one with vertical lines at fractional
        positions *f1*, *f2*, ...
        """
        boxes = []
        xf = [0] + list(args) + [1]
        x0, y0, x1, y1 = self._get_extents()
        w = x1 - x0
        for xf0, xf1 in zip(xf[:-1], xf[1:]):
            boxes.append(Bbox([[x0 + xf0 * w, y0], [x0 + xf1 * w, y1]]))
        return boxes

    def splity(self, *args):
        """
        e.g., ``bbox.splitx(f1, f2, ...)``

        Returns a list of new :class:`Bbox` objects formed by
        splitting the original one with horizontal lines at fractional
        positions *f1*, *f2*, ...
        """
        boxes = []
        yf = [0] + list(args) + [1]
        x0, y0, x1, y1 = self._get_extents()
        h = y1 - y0
        for yf0, yf1 in zip(yf[:-1], yf[1:]):
            boxes.append(Bbox([[x0, y0 + yf0 * h], [x1, y0 + yf1 * h]]))
        return boxes

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the :class:`Bbox`.

        *vertices* is a Nx2 Numpy array.
        """
        if len(vertices) == 0:
            return 0
        vertices = np.asarray(vertices)
        x0, y0, x1, y1 = self._get_extents()
        dx0 = np.sign(vertices[:, 0] - x0)
        dy0 = np.sign(vertices[:, 1] - y0)
        dx1 = np.sign(vertices[:, 0] - x1)
        dy1 = np.sign(vertices[:, 1] - y1)
        inside = ((abs(dx0 + dx1) + abs(dy0 + dy1)) == 0)
        return np.sum(inside)

    def count_overlaps(self, bboxes):
        """
        Count the number of bounding boxes that overlap this one.

        bboxes is a sequence of :class:`BboxBase` objects
        """
        return count_bboxes_overlapping_bbox(self, bboxes)

    def expanded(self, sw, sh):
        """
        Return a new :class:`Bbox` which is this :class:`Bbox`
        expanded around its center by the given factors *sw* and
        *sh*.
        """
        width = self.width
        height = self.height
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        a = np.array([[-deltaw, -deltah], [deltaw, deltah]])
        return Bbox(self._points + a)

    def padded(self, p):
        """
        Return a new :class:`Bbox` that is padded on all four sides by
        the given value.
        """
        points = self.get_points()
        return Bbox(points + [[-p, -p], [p, p]])

    def translated(self, tx, ty):
        """
        Return a copy of the :class:`Bbox`, statically translated by
        *tx* and *ty*.
        """
        return Bbox(self._points + (tx, ty))

    def corners(self):
        """
        Return an array of points which are the four corners of this
        rectangle.  For example, if this :class:`Bbox` is defined by
        the points (*a*, *b*) and (*c*, *d*), :meth:`corners` returns
        (*a*, *b*), (*a*, *d*), (*c*, *b*) and (*c*, *d*).
        """
        l, b, r, t = self.get_points().flatten()
        return np.array([[l, b], [l, t], [r, b], [r, t]])

    def rotated(self, radians):
        """
        Return a new bounding box that bounds a rotated version of
        this bounding box by the given radians.  The new bounding box
        is still aligned with the axes, of course.
        """
        corners = self.corners()
        corners_rotated = Affine2D().rotate(radians).transform(corners)
        bbox = Bbox.unit()
        bbox.update_from_data_xy(corners_rotated, ignore=True)
        return bbox

    @staticmethod
    def union(bboxes):
        """
        Return a :class:`Bbox` that contains all of the given bboxes.
        """
        assert(len(bboxes))

        if len(bboxes) == 1:
            return bboxes[0]

        x0 = np.inf
        y0 = np.inf
        x1 = -np.inf
        y1 = -np.inf

        for bbox in bboxes:
            points = bbox.get_points()
            xs = points[:, 0]
            ys = points[:, 1]
            x0 = min(x0, np.min(xs))
            y0 = min(y0, np.min(ys))
            x1 = max(x1, np.max(xs))
            y1 = max(y1, np.max(ys))

        return Bbox.from_extents(x0, y0, x1, y1)

    @staticmethod
    def intersection(bbox1, bbox2):
        """
        Return the intersection of the two bboxes or None
        if they do not intersect.

        Implements the algorithm described at:

            http://www.tekpool.com/node/2687

        """
        intersects = not (bbox2.xmin > bbox1.xmax or
                          bbox2.xmax < bbox1.xmin or
                          bbox2.ymin > bbox1.ymax or
                          bbox2.ymax < bbox1.ymin)

        if intersects:
            x0 = max([bbox1.xmin, bbox2.xmin])
            x1 = min([bbox1.xmax, bbox2.xmax])
            y0 = max([bbox1.ymin, bbox2.ymin])
            y1 = min([bbox1.ymax, bbox2.ymax])
            return Bbox.from_extents(x0, y0, x1, y1)

        return None


class Bbox(BboxBase):
    """
    A mutable bounding box.
    """

    def __init__(self, points, **kwargs):
        """
        *points*: a 2x2 numpy array of the form [[x0, y0], [x1, y1]]

        If you need to create a :class:`Bbox` object from another form
        of data, consider the static methods :meth:`unit`,
        :meth:`from_bounds` and :meth:`from_extents`.
        """
        BboxBase.__init__(self, **kwargs)
        points = np.asarray(points, np.float_)
        if points.shape != (2, 2):
            raise ValueError('Bbox points must be of the form '
                             '"[[x0, y0], [x1, y1]]".')
        self._points = points
        self._minpos = np.array([0.0000001, 0.0000001])
        self._ignore = True
        # it is helpful in some contexts to know if the bbox is a
        # default or has been mutated; we store the orig points to
        # support the mutated methods
        self._points_orig = self._points.copy()
    if DEBUG:
        ___init__ = __init__

        def __init__(self, points, **kwargs):
            self._check(points)
            self.___init__(points, **kwargs)

        def invalidate(self):
            self._check(self._points)
            TransformNode.invalidate(self)

    @staticmethod
    def unit():
        """
        (staticmethod) Create a new unit :class:`Bbox` from (0, 0) to
        (1, 1).
        """
        return Bbox(np.array([[0.0, 0.0], [1.0, 1.0]], np.float))

    @staticmethod
    def null():
        """
        (staticmethod) Create a new null :class:`Bbox` from (inf, inf) to
        (-inf, -inf).
        """
        return Bbox(np.array([[np.inf, np.inf], [-np.inf, -np.inf]], np.float))

    @staticmethod
    def from_bounds(x0, y0, width, height):
        """
        (staticmethod) Create a new :class:`Bbox` from *x0*, *y0*,
        *width* and *height*.

        *width* and *height* may be negative.
        """
        return Bbox.from_extents(x0, y0, x0 + width, y0 + height)

    @staticmethod
    def from_extents(*args):
        """
        (staticmethod) Create a new Bbox from *left*, *bottom*,
        *right* and *top*.

        The *y*-axis increases upwards.
        """
        points = np.array(args, dtype=np.float_).reshape(2, 2)
        return Bbox(points)

    def __repr__(self):
        return 'Bbox(%r)' % repr(self._points)

    def ignore(self, value):
        """
        Set whether the existing bounds of the box should be ignored
        by subsequent calls to :meth:`update_from_data` or
        :meth:`update_from_data_xy`.

        *value*:

           - When True, subsequent calls to :meth:`update_from_data`
             will ignore the existing bounds of the :class:`Bbox`.

           - When False, subsequent calls to :meth:`update_from_data`
             will include the existing bounds of the :class:`Bbox`.
        """
        self._ignore = value

    def update_from_data(self, x, y, ignore=None):
        """
        Update the bounds of the :class:`Bbox` based on the passed in
        data.  After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        *x*: a numpy array of *x*-values

        *y*: a numpy array of *y*-values

        *ignore*:
           - when True, ignore the existing bounds of the :class:`Bbox`.
           - when False, include the existing bounds of the :class:`Bbox`.
           - when None, use the last value passed to :meth:`ignore`.
        """
        warnings.warn(
            "update_from_data requires a memory copy -- please replace with "
            "update_from_data_xy")

        xy = np.hstack((x.reshape((len(x), 1)), y.reshape((len(y), 1))))
        return self.update_from_data_xy(xy, ignore)

    def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
        """
        Update the bounds of the :class:`Bbox` based on the passed in
        data.  After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        *path*: a :class:`~matplotlib.path.Path` instance

        *ignore*:
           - when True, ignore the existing bounds of the :class:`Bbox`.
           - when False, include the existing bounds of the :class:`Bbox`.
           - when None, use the last value passed to :meth:`ignore`.

        *updatex*: when True, update the x values

        *updatey*: when True, update the y values

        """
        if ignore is None:
            ignore = self._ignore

        if path.vertices.size == 0:
            return

        points, minpos, changed = update_path_extents(
            path, None, self._points, self._minpos, ignore)

        if changed:
            self.invalidate()
            if updatex:
                self._points[:, 0] = points[:, 0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:, 1] = points[:, 1]
                self._minpos[1] = minpos[1]

    def update_from_data_xy(self, xy, ignore=None, updatex=True, updatey=True):
        """
        Update the bounds of the :class:`Bbox` based on the passed in
        data.  After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        *xy*: a numpy array of 2D points

        *ignore*:
           - when True, ignore the existing bounds of the :class:`Bbox`.
           - when False, include the existing bounds of the :class:`Bbox`.
           - when None, use the last value passed to :meth:`ignore`.

        *updatex*: when True, update the x values

        *updatey*: when True, update the y values
        """
        if len(xy) == 0:
            return

        path = Path(xy)
        self.update_from_path(path, ignore=ignore,
                                    updatex=updatex, updatey=updatey)

    def _set_x0(self, val):
        self._points[0, 0] = val
        self.invalidate()
    x0 = property(BboxBase._get_x0, _set_x0)

    def _set_y0(self, val):
        self._points[0, 1] = val
        self.invalidate()
    y0 = property(BboxBase._get_y0, _set_y0)

    def _set_x1(self, val):
        self._points[1, 0] = val
        self.invalidate()
    x1 = property(BboxBase._get_x1, _set_x1)

    def _set_y1(self, val):
        self._points[1, 1] = val
        self.invalidate()
    y1 = property(BboxBase._get_y1, _set_y1)

    def _set_p0(self, val):
        self._points[0] = val
        self.invalidate()
    p0 = property(BboxBase._get_p0, _set_p0)

    def _set_p1(self, val):
        self._points[1] = val
        self.invalidate()
    p1 = property(BboxBase._get_p1, _set_p1)

    def _set_intervalx(self, interval):
        self._points[:, 0] = interval
        self.invalidate()
    intervalx = property(BboxBase._get_intervalx, _set_intervalx)

    def _set_intervaly(self, interval):
        self._points[:, 1] = interval
        self.invalidate()
    intervaly = property(BboxBase._get_intervaly, _set_intervaly)

    def _set_bounds(self, bounds):
        l, b, w, h = bounds
        points = np.array([[l, b], [l + w, b + h]], np.float_)
        if np.any(self._points != points):
            self._points = points
            self.invalidate()
    bounds = property(BboxBase._get_bounds, _set_bounds)

    def _get_minpos(self):
        return self._minpos
    minpos = property(_get_minpos)

    def _get_minposx(self):
        return self._minpos[0]
    minposx = property(_get_minposx)

    def _get_minposy(self):
        return self._minpos[1]
    minposy = property(_get_minposy)

    def get_points(self):
        """
        Get the points of the bounding box directly as a numpy array
        of the form: [[x0, y0], [x1, y1]].
        """
        self._invalid = 0
        return self._points

    def set_points(self, points):
        """
        Set the points of the bounding box directly from a numpy array
        of the form: [[x0, y0], [x1, y1]].  No error checking is
        performed, as this method is mainly for internal use.
        """
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    def set(self, other):
        """
        Set this bounding box from the "frozen" bounds of another
        :class:`Bbox`.
        """
        if np.any(self._points != other.get_points()):
            self._points = other.get_points()
            self.invalidate()

    def mutated(self):
        'return whether the bbox has changed since init'
        return self.mutatedx() or self.mutatedy()

    def mutatedx(self):
        'return whether the x-limits have changed since init'
        return (self._points[0, 0] != self._points_orig[0, 0] or
                self._points[1, 0] != self._points_orig[1, 0])

    def mutatedy(self):
        'return whether the y-limits have changed since init'
        return (self._points[0, 1] != self._points_orig[0, 1] or
                self._points[1, 1] != self._points_orig[1, 1])


class TransformedBbox(BboxBase):
    """
    A :class:`Bbox` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """
    def __init__(self, bbox, transform, **kwargs):
        """
        *bbox*: a child :class:`Bbox`

        *transform*: a 2D :class:`Transform`
        """
        assert bbox.is_bbox
        assert isinstance(transform, Transform)
        assert transform.input_dims == 2
        assert transform.output_dims == 2

        BboxBase.__init__(self, **kwargs)
        self._bbox = bbox
        self._transform = transform
        self.set_children(bbox, transform)
        self._points = None

    def __repr__(self):
        return "TransformedBbox(%r, %r)" % (self._bbox, self._transform)

    def get_points(self):
        if self._invalid:
            points = self._transform.transform(self._bbox.get_points())
            points = np.ma.filled(points, 0.0)
            self._points = points
            self._invalid = 0
        return self._points
    get_points.__doc__ = Bbox.get_points.__doc__

    if DEBUG:
        _get_points = get_points

        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points


class Transform(TransformNode):
    """
    The base class of all :class:`TransformNode` instances that
    actually perform a transformation.

    All non-affine transformations should be subclasses of this class.
    New affine transformations should be subclasses of
    :class:`Affine2D`.

    Subclasses of this class should override the following members (at
    minimum):

      - :attr:`input_dims`
      - :attr:`output_dims`
      - :meth:`transform`
      - :attr:`is_separable`
      - :attr:`has_inverse`
      - :meth:`inverted` (if :attr:`has_inverse` is True)

    If the transform needs to do something non-standard with
    :class:`matplotlib.path.Path` objects, such as adding curves
    where there were once line segments, it should override:

      - :meth:`transform_path`
    """
    input_dims = None
    """
    The number of input dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    output_dims = None
    """
    The number of output dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    has_inverse = False
    """True if this transform has a corresponding inverse transform."""

    is_separable = False
    """True if this transform is separable in the x- and y- dimensions."""

    def __add__(self, other):
        """
        Composes two transforms together such that *self* is followed
        by *other*.
        """
        if isinstance(other, Transform):
            return composite_transform_factory(self, other)
        raise TypeError(
            "Can not add Transform to object of type '%s'" % type(other))

    def __radd__(self, other):
        """
        Composes two transforms together such that *self* is followed
        by *other*.
        """
        if isinstance(other, Transform):
            return composite_transform_factory(other, self)
        raise TypeError(
            "Can not add Transform to object of type '%s'" % type(other))

    def __eq__(self, other):
        # equality is based on transform object id. Hence:
        # Transform() != Transform().
        # Some classes, such as TransformWrapper & AffineBase, will override.
        return self is other

    def _iter_break_from_left_to_right(self):
        """
        Returns an iterator breaking down this transform stack from left to
        right recursively. If self == ((A, N), A) then the result will be an
        iterator which yields I : ((A, N), A), followed by A : (N, A),
        followed by (A, N) : (A), but not ((A, N), A) : I.

        This is equivalent to flattening the stack then yielding
        ``flat_stack[:i], flat_stack[i:]`` where i=0..(n-1).

        """
        yield IdentityTransform(), self

    @property
    def depth(self):
        """
        Returns the number of transforms which have been chained
        together to form this Transform instance.

        .. note::

            For the special case of a Composite transform, the maximum depth
            of the two is returned.

        """
        return 1

    def contains_branch(self, other):
        """
        Return whether the given transform is a sub-tree of this transform.

        This routine uses transform equality to identify sub-trees, therefore
        in many situations it is object id which will be used.

        For the case where the given transform represents the whole
        of this transform, returns True.

        """
        if self.depth < other.depth:
            return False

        # check that a subtree is equal to other (starting from self)
        for _, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return True
        return False

    def contains_branch_seperately(self, other_transform):
        """
        Returns whether the given branch is a sub-tree of this transform on
        each seperate dimension.

        A common use for this method is to identify if a transform is a blended
        transform containing an axes' data transform. e.g.::

            x_isdata, y_isdata = trans.contains_branch_seperately(ax.transData)

        """
        if self.output_dims != 2:
            raise ValueError('contains_branch_seperately only supports '
                             'transforms with 2 output dimensions')
        # for a non-blended transform each seperate dimension is the same, so
        # just return the appropriate shape.
        return [self.contains_branch(other_transform)] * 2

    def __sub__(self, other):
        """
        Returns a transform stack which goes all the way down self's transform
        stack, and then ascends back up other's stack. If it can, this is
        optimised::

            # normally
            A - B == a + b.inverted()

            # sometimes, when A contains the tree B there is no need to
            # descend all the way down to the base of A (via B), instead we
            # can just stop at B.

            (A + B) - (B)^-1 == A

            # similarly, when B contains tree A, we can avoid decending A at
            # all, basically:
            A - (A + B) == ((B + A) - A).inverted() or B^-1

        For clarity, the result of ``(A + B) - B + B == (A + B)``.

        """
        # we only know how to do this operation if other is a Transform.
        if not isinstance(other, Transform):
            return NotImplemented

        for remainder, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return remainder

        for remainder, sub_tree in other._iter_break_from_left_to_right():
            if sub_tree == self:
                if not remainder.has_inverse:
                    raise ValueError("The shortcut cannot be computed since "
                     "other's transform includes a non-invertable component.")
                return remainder.inverted()

        # if we have got this far, then there was no shortcut possible
        if other.has_inverse:
            return self + other.inverted()
        else:
            raise ValueError('It is not possible to compute transA - transB '
                             'since transB cannot be inverted and there is no '
                             'shortcut possible.')

    def __array__(self, *args, **kwargs):
        """
        Array interface to get at this Transform's affine matrix.
        """
        return self.get_affine().get_matrix()

    def transform(self, values):
        """
        Performs the transformation on the given array of values.

        Accepts a numpy array of shape (N x :attr:`input_dims`) and
        returns a numpy array of shape (N x :attr:`output_dims`).
        """
        return self.transform_affine(self.transform_non_affine(values))

    def transform_affine(self, values):
        """
        Performs only the affine part of this transformation on the
        given array of values.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally a no-op.  In
        affine transformations, this is equivalent to
        ``transform(values)``.

        Accepts a numpy array of shape (N x :attr:`input_dims`) and
        returns a numpy array of shape (N x :attr:`output_dims`).
        """
        return self.get_affine().transform(values)

    def transform_non_affine(self, values):
        """
        Performs only the non-affine part of the transformation.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally equivalent to
        ``transform(values)``.  In affine transformations, this is
        always a no-op.

        Accepts a numpy array of shape (N x :attr:`input_dims`) and
        returns a numpy array of shape (N x :attr:`output_dims`).
        """
        return values

    def get_affine(self):
        """
        Get the affine part of this transform.
        """
        return IdentityTransform()

    def get_matrix(self):
        """
        Get the Affine transformation array for the affine part
        of this transform.

        """
        return self.get_affine().get_matrix()

    def transform_point(self, point):
        """
        A convenience function that returns the transformed copy of a
        single point.

        The point is given as a sequence of length :attr:`input_dims`.
        The transformed point is returned as a sequence of length
        :attr:`output_dims`.
        """
        assert len(point) == self.input_dims
        return self.transform(np.asarray([point]))[0]

    def transform_path(self, path):
        """
        Returns a transformed path.

        *path*: a :class:`~matplotlib.path.Path` instance.

        In some cases, this transform may insert curves into the path
        that began as line segments.
        """
        return self.transform_path_affine(self.transform_path_non_affine(path))

    def transform_path_affine(self, path):
        """
        Returns a path, transformed only by the affine part of
        this transform.

        *path*: a :class:`~matplotlib.path.Path` instance.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        return self.get_affine().transform_path_affine(path)

    def transform_path_non_affine(self, path):
        """
        Returns a path, transformed only by the non-affine
        part of this transform.

        *path*: a :class:`~matplotlib.path.Path` instance.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        return Path(self.transform_non_affine(path.vertices), path.codes,
                    path._interpolation_steps)

    def transform_angles(self, angles, pts, radians=False, pushoff=1e-5):
        """
        Performs transformation on a set of angles anchored at
        specific locations.

        The *angles* must be a column vector (i.e., numpy array).

        The *pts* must be a two-column numpy array of x,y positions
        (angle transforms currently only work in 2D).  This array must
        have the same number of rows as *angles*.

        *radians* indicates whether or not input angles are given in
         radians (True) or degrees (False; the default).

        *pushoff* is the distance to move away from *pts* for
         determining transformed angles (see discussion of method
         below).

        The transformed angles are returned in an array with the same
        size as *angles*.

        The generic version of this method uses a very generic
        algorithm that transforms *pts*, as well as locations very
        close to *pts*, to find the angle in the transformed system.
        """
        # Must be 2D
        if self.input_dims != 2 or self.output_dims != 2:
            raise NotImplementedError('Only defined in 2D')

        # pts must be array with 2 columns for x,y
        assert pts.shape[1] == 2

        # angles must be a column vector and have same number of
        # rows as pts
        assert np.prod(angles.shape) == angles.shape[0] == pts.shape[0]

        # Convert to radians if desired
        if not radians:
            angles = angles / 180.0 * np.pi

        # Move a short distance away
        pts2 = pts + pushoff * np.c_[np.cos(angles), np.sin(angles)]

        # Transform both sets of points
        tpts = self.transform(pts)
        tpts2 = self.transform(pts2)

        # Calculate transformed angles
        d = tpts2 - tpts
        a = np.arctan2(d[:, 1], d[:, 0])

        # Convert back to degrees if desired
        if not radians:
            a = a * 180.0 / np.pi

        return a

    def inverted(self):
        """
        Return the corresponding inverse transformation.

        The return value of this method should be treated as
        temporary.  An update to *self* does not cause a corresponding
        update to its inverted copy.

        ``x === self.inverted().transform(self.transform(x))``
        """
        raise NotImplementedError()


class TransformWrapper(Transform):
    """
    A helper class that holds a single child transform and acts
    equivalently to it.

    This is useful if a node of the transform tree must be replaced at
    run time with a transform of a different type.  This class allows
    that replacement to correctly trigger invalidation.

    Note that :class:`TransformWrapper` instances must have the same
    input and output dimensions during their entire lifetime, so the
    child transform may only be replaced with another child transform
    of the same dimensions.
    """
    pass_through = True

    def __init__(self, child):
        """
        *child*: A class:`Transform` instance.  This child may later
        be replaced with :meth:`set`.
        """
        assert isinstance(child, Transform)
        Transform.__init__(self)
        self.input_dims = child.input_dims
        self.output_dims = child.output_dims
        self._set(child)
        self._invalid = 0

    def __eq__(self, other):
        return self._child.__eq__(other)

    if DEBUG:

        def __str__(self):
            return str(self._child)

    def __getstate__(self):
        # only store the child
        return {'child': self._child}

    def __setstate__(self, state):
        # re-initialise the TransformWrapper with the state's child
        self.__init__(state['child'])

    def __repr__(self):
        return "TransformWrapper(%r)" % self._child

    def frozen(self):
        return self._child.frozen()
    frozen.__doc__ = Transform.frozen.__doc__

    def _set(self, child):
        self._child = child
        self.set_children(child)

        self.transform = child.transform
        self.transform_affine = child.transform_affine
        self.transform_non_affine = child.transform_non_affine
        self.transform_path = child.transform_path
        self.transform_path_affine = child.transform_path_affine
        self.transform_path_non_affine = child.transform_path_non_affine
        self.get_affine = child.get_affine
        self.inverted = child.inverted
        self.get_matrix = child.get_matrix

        # note we do not wrap other properties here since the transform's
        # child can be changed with WrappedTransform.set and so checking
        # is_affine and other such properties may be dangerous.

    def set(self, child):
        """
        Replace the current child of this transform with another one.

        The new child must have the same number of input and output
        dimensions as the current child.
        """
        assert child.input_dims == self.input_dims
        assert child.output_dims == self.output_dims

        self._set(child)

        self._invalid = 0
        self.invalidate()
        self._invalid = 0

    def _get_is_affine(self):
        return self._child.is_affine
    is_affine = property(_get_is_affine)

    def _get_is_separable(self):
        return self._child.is_separable
    is_separable = property(_get_is_separable)

    def _get_has_inverse(self):
        return self._child.has_inverse
    has_inverse = property(_get_has_inverse)


class AffineBase(Transform):
    """
    The base class of all affine transformations of any number of
    dimensions.
    """
    is_affine = True

    def __init__(self, *args, **kwargs):
        Transform.__init__(self, *args, **kwargs)
        self._inverted = None

    def __array__(self, *args, **kwargs):
        # optimises the access of the transform matrix vs the superclass
        return self.get_matrix()

    @staticmethod
    def _concat(a, b):
        """
        Concatenates two transformation matrices (represented as numpy
        arrays) together.
        """
        return np.dot(b, a)

    def __eq__(self, other):
        if other.is_affine:
            return np.all(self.get_matrix() == other.get_matrix())
        return NotImplemented

    def transform(self, values):
        return self.transform_affine(values)
    transform.__doc__ = Transform.transform.__doc__

    def transform_affine(self, values):
        raise NotImplementedError('Affine subclasses should override this '
                                  'method.')
    transform_affine.__doc__ = Transform.transform_affine.__doc__

    def transform_non_affine(self, points):
        return points
    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def transform_path(self, path):
        return self.transform_path_affine(path)
    transform_path.__doc__ = Transform.transform_path.__doc__

    def transform_path_affine(self, path):
        return Path(self.transform_affine(path.vertices),
                    path.codes, path._interpolation_steps)
    transform_path_affine.__doc__ = Transform.transform_path_affine.__doc__

    def transform_path_non_affine(self, path):
        return path
    transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__

    def get_affine(self):
        return self
    get_affine.__doc__ = Transform.get_affine.__doc__


class Affine2DBase(AffineBase):
    """
    The base class of all 2D affine transformations.

    2D affine transformations are performed using a 3x3 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 2D
    affine transformation, use :class:`Affine2D`.

    Subclasses of this class will generally only need to override a
    constructor and :meth:`get_matrix` that generates a custom 3x3 matrix.
    """
    has_inverse = True

    input_dims = 2
    output_dims = 2

    def frozen(self):
        return Affine2D(self.get_matrix().copy())
    frozen.__doc__ = AffineBase.frozen.__doc__

    def _get_is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0
    is_separable = property(_get_is_separable)

    def to_values(self):
        """
        Return the values of the matrix as a sequence (a,b,c,d,e,f)
        """
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flatten())

    @staticmethod
    def matrix_from_values(a, b, c, d, e, f):
        """
        (staticmethod) Create a new transformation matrix as a 3x3
        numpy array of the form::

          a c e
          b d f
          0 0 1
        """
        return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], np.float_)

    def transform_affine(self, points):
        mtx = self.get_matrix()
        if isinstance(points, MaskedArray):
            tpoints = affine_transform(points.data, mtx)
            return ma.MaskedArray(tpoints, mask=ma.getmask(points))
        return affine_transform(points, mtx)

    def transform_point(self, point):
        mtx = self.get_matrix()
        return affine_transform(point, mtx)
    transform_point.__doc__ = AffineBase.transform_point.__doc__

    if DEBUG:
        _transform_affine = transform_affine

        def transform_affine(self, points):
            # The major speed trap here is just converting to the
            # points to an array in the first place.  If we can use
            # more arrays upstream, that should help here.
            if (not ma.isMaskedArray(points) and
                not isinstance(points, np.ndarray)):
                warnings.warn(
                    ('A non-numpy array of type %s was passed in for ' +
                     'transformation.  Please correct this.')
                    % type(points))
            return self._transform_affine(points)
    transform_affine.__doc__ = AffineBase.transform_affine.__doc__

    def inverted(self):
        if self._inverted is None or self._invalid:
            mtx = self.get_matrix()
            shorthand_name = None
            if self._shorthand_name:
                shorthand_name = '(%s)-1' % self._shorthand_name
            self._inverted = Affine2D(inv(mtx), shorthand_name=shorthand_name)
            self._invalid = 0
        return self._inverted
    inverted.__doc__ = AffineBase.inverted.__doc__


class Affine2D(Affine2DBase):
    """
    A mutable 2D affine transformation.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Initialize an Affine transform from a 3x3 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        Affine2DBase.__init__(self, **kwargs)
        if matrix is None:
            matrix = np.identity(3)
        elif DEBUG:
            matrix = np.asarray(matrix, np.float_)
            assert matrix.shape == (3, 3)
        self._mtx = matrix
        self._invalid = 0

    def __repr__(self):
        return "Affine2D(%s)" % repr(self._mtx)

#    def __cmp__(self, other):
#        # XXX redundant. this only tells us eq.
#        if (isinstance(other, Affine2D) and
#            (self.get_matrix() == other.get_matrix()).all()):
#            return 0
#        return -1

    @staticmethod
    def from_values(a, b, c, d, e, f):
        """
        (staticmethod) Create a new Affine2D instance from the given
        values::

          a c e
          b d f
          0 0 1

        .
        """
        return Affine2D(
            np.array([a, c, e, b, d, f, 0.0, 0.0, 1.0], np.float_)
            .reshape((3, 3)))

    def get_matrix(self):
        """
        Get the underlying transformation matrix as a 3x3 numpy array::

          a c e
          b d f
          0 0 1

        .
        """
        self._invalid = 0
        return self._mtx

    def set_matrix(self, mtx):
        """
        Set the underlying transformation matrix from a 3x3 numpy array::

          a c e
          b d f
          0 0 1

        .
        """
        self._mtx = mtx
        self.invalidate()

    def set(self, other):
        """
        Set this transformation from the frozen copy of another
        :class:`Affine2DBase` object.
        """
        assert isinstance(other, Affine2DBase)
        self._mtx = other.get_matrix()
        self.invalidate()

    @staticmethod
    def identity():
        """
        (staticmethod) Return a new :class:`Affine2D` object that is
        the identity transform.

        Unless this transform will be mutated later on, consider using
        the faster :class:`IdentityTransform` class instead.
        """
        return Affine2D(np.identity(3))

    def clear(self):
        """
        Reset the underlying matrix to the identity transform.
        """
        self._mtx = np.identity(3)
        self.invalidate()
        return self

    def rotate(self, theta):
        """
        Add a rotation (in radians) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        a = np.cos(theta)
        b = np.sin(theta)
        rotate_mtx = np.array(
            [[a, -b, 0.0], [b, a, 0.0], [0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(rotate_mtx, self._mtx)
        self.invalidate()
        return self

    def rotate_deg(self, degrees):
        """
        Add a rotation (in degrees) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.rotate(degrees * np.pi / 180.)

    def rotate_around(self, x, y, theta):
        """
        Add a rotation (in radians) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """
        Add a rotation (in degrees) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)

    def translate(self, tx, ty):
        """
        Adds a translation in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        translate_mtx = np.array(
            [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(translate_mtx, self._mtx)
        self.invalidate()
        return self

    def scale(self, sx, sy=None):
        """
        Adds a scale in place.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        if sy is None:
            sy = sx
        scale_mtx = np.array(
            [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
            np.float_)
        self._mtx = np.dot(scale_mtx, self._mtx)
        self.invalidate()
        return self

    def _get_is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0
    is_separable = property(_get_is_separable)


class IdentityTransform(Affine2DBase):
    """
    A special class that does on thing, the identity transform, in a
    fast way.
    """
    _mtx = np.identity(3)

    def frozen(self):
        return self
    frozen.__doc__ = Affine2DBase.frozen.__doc__

    def __repr__(self):
        return "IdentityTransform()"

    def get_matrix(self):
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__

    def transform(self, points):
        return points
    transform.__doc__ = Affine2DBase.transform.__doc__

    transform_affine = transform
    transform_affine.__doc__ = Affine2DBase.transform_affine.__doc__

    transform_non_affine = transform
    transform_non_affine.__doc__ = Affine2DBase.transform_non_affine.__doc__

    def transform_path(self, path):
        return path
    transform_path.__doc__ = Affine2DBase.transform_path.__doc__

    transform_path_affine = transform_path
    transform_path_affine.__doc__ = Affine2DBase.transform_path_affine.__doc__

    transform_path_non_affine = transform_path
    transform_path_non_affine.__doc__ = Affine2DBase.transform_path_non_affine.__doc__

    def get_affine(self):
        return self
    get_affine.__doc__ = Affine2DBase.get_affine.__doc__

    inverted = get_affine
    inverted.__doc__ = Affine2DBase.inverted.__doc__


class BlendedGenericTransform(Transform):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This "generic" version can handle any given child transform in the
    *x*- and *y*-directions.
    """
    input_dims = 2
    output_dims = 2
    is_separable = True
    pass_through = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to
        transform the *x*-axis and *y_transform* to transform the
        *y*-axis.

        You will generally not call this constructor directly but use
        the :func:`blended_transform_factory` function instead, which
        can determine automatically which kind of blended transform to
        create.
        """
        # Here we ask: "Does it blend?"

        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)
        self._affine = None

    def __eq__(self, other):
        # Note, this is an exact copy of BlendedAffine2D.__eq__
        if isinstance(other, (BlendedAffine2D, BlendedGenericTransform)):
            return (self._x == other._x) and (self._y == other._y)
        elif self._x == self._y:
            return self._x == other
        else:
            return NotImplemented

    def contains_branch_seperately(self, transform):
        # Note, this is an exact copy of BlendedAffine2D.contains_branch_seperately
        return self._x.contains_branch(transform), self._y.contains_branch(transform)

    @property
    def depth(self):
        return max([self._x.depth, self._y.depth])

    def contains_branch(self, other):
        # a blended transform cannot possibly contain a branch from two different transforms.
        return False

    def _get_is_affine(self):
        return self._x.is_affine and self._y.is_affine
    is_affine = property(_get_is_affine)

    def _get_has_inverse(self):
        return self._x.has_inverse and self._y.has_inverse
    has_inverse = property(_get_has_inverse)

    def frozen(self):
        return blended_transform_factory(self._x.frozen(), self._y.frozen())
    frozen.__doc__ = Transform.frozen.__doc__

    def __repr__(self):
        return "BlendedGenericTransform(%s,%s)" % (self._x, self._y)

    def transform_non_affine(self, points):
        if self._x.is_affine and self._y.is_affine:
            return points
        x = self._x
        y = self._y

        if x == y and x.input_dims == 2:
            return x.transform_non_affine(points)

        if x.input_dims == 2:
            x_points = x.transform_non_affine(points)[:, 0:1]
        else:
            x_points = x.transform_non_affine(points[:, 0])
            x_points = x_points.reshape((len(x_points), 1))

        if y.input_dims == 2:
            y_points = y.transform_non_affine(points)[:, 1:]
        else:
            y_points = y.transform_non_affine(points[:, 1])
            y_points = y_points.reshape((len(y_points), 1))

        if isinstance(x_points, MaskedArray) or isinstance(y_points, MaskedArray):
            return ma.concatenate((x_points, y_points), 1)
        else:
            return np.concatenate((x_points, y_points), 1)
    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def inverted(self):
        return BlendedGenericTransform(self._x.inverted(), self._y.inverted())
    inverted.__doc__ = Transform.inverted.__doc__

    def get_affine(self):
        if self._invalid or self._affine is None:
            if self._x == self._y:
                self._affine = self._x.get_affine()
            else:
                x_mtx = self._x.get_affine().get_matrix()
                y_mtx = self._y.get_affine().get_matrix()
                # This works because we already know the transforms are
                # separable, though normally one would want to set b and
                # c to zero.
                mtx = np.vstack((x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]))
                self._affine = Affine2D(mtx)
            self._invalid = 0
        return self._affine
    get_affine.__doc__ = Transform.get_affine.__doc__


class BlendedAffine2D(Affine2DBase):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This version is an optimization for the case where both child
    transforms are of type :class:`Affine2DBase`.
    """
    is_separable = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to
        transform the *x*-axis and *y_transform* to transform the
        *y*-axis.

        Both *x_transform* and *y_transform* must be 2D affine
        transforms.

        You will generally not call this constructor directly but use
        the :func:`blended_transform_factory` function instead, which
        can determine automatically which kind of blended transform to
        create.
        """
        assert x_transform.is_affine
        assert y_transform.is_affine
        assert x_transform.is_separable
        assert y_transform.is_separable

        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)

        Affine2DBase.__init__(self)
        self._mtx = None

    def __eq__(self, other):
        # Note, this is an exact copy of BlendedGenericTransform.__eq__
        if isinstance(other, (BlendedAffine2D, BlendedGenericTransform)):
            return (self._x == other._x) and (self._y == other._y)
        elif self._x == self._y:
            return self._x == other
        else:
            return NotImplemented

    def contains_branch_seperately(self, transform):
        # Note, this is an exact copy of BlendedTransform.contains_branch_seperately
        return self._x.contains_branch(transform), self._y.contains_branch(transform)

    def __repr__(self):
        return "BlendedAffine2D(%s,%s)" % (self._x, self._y)

    def get_matrix(self):
        if self._invalid:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()
            else:
                x_mtx = self._x.get_matrix()
                y_mtx = self._y.get_matrix()
                # This works because we already know the transforms are
                # separable, though normally one would want to set b and
                # c to zero.
                self._mtx = np.vstack((x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]))
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


def blended_transform_factory(x_transform, y_transform):
    """
    Create a new "blended" transform using *x_transform* to transform
    the *x*-axis and *y_transform* to transform the *y*-axis.

    A faster version of the blended transform is returned for the case
    where both child transforms are affine.
    """
    if (isinstance(x_transform, Affine2DBase)
        and isinstance(y_transform, Affine2DBase)):
        return BlendedAffine2D(x_transform, y_transform)
    return BlendedGenericTransform(x_transform, y_transform)


class CompositeGenericTransform(Transform):
    """
    A composite transform formed by applying transform *a* then
    transform *b*.

    This "generic" version can handle any two arbitrary
    transformations.
    """
    pass_through = True

    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        You will generally not call this constructor directly but use
        the :func:`composite_transform_factory` function instead,
        which can automatically choose the best kind of composite
        transform instance to create.
        """
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        Transform.__init__(self, **kwargs)
        self._a = a
        self._b = b
        self.set_children(a, b)

    is_affine = property(lambda self: self._a.is_affine and self._b.is_affine)

    def frozen(self):
        self._invalid = 0
        frozen = composite_transform_factory(self._a.frozen(), self._b.frozen())
        if not isinstance(frozen, CompositeGenericTransform):
            return frozen.frozen()
        return frozen
    frozen.__doc__ = Transform.frozen.__doc__

    def _invalidate_internal(self, value, invalidating_node):
        # In some cases for a composite transform, an invalidating call to AFFINE_ONLY needs
        # to be extended to invalidate the NON_AFFINE part too. These cases are when the right
        # hand transform is non-affine and either:
        # (a) the left hand transform is non affine
        # (b) it is the left hand node which has triggered the invalidation
        if value == Transform.INVALID_AFFINE \
            and not self._b.is_affine \
            and (not self._a.is_affine or invalidating_node is self._a):

            value = Transform.INVALID

        Transform._invalidate_internal(self, value=value,
                                       invalidating_node=invalidating_node)

    def __eq__(self, other):
        if isinstance(other, (CompositeGenericTransform, CompositeAffine2D)):
            return self is other or (self._a == other._a and self._b == other._b)
        else:
            return False

    def _iter_break_from_left_to_right(self):
        for lh_compliment, rh_compliment in self._a._iter_break_from_left_to_right():
            yield lh_compliment, rh_compliment + self._b
        for lh_compliment, rh_compliment in self._b._iter_break_from_left_to_right():
            yield self._a + lh_compliment, rh_compliment

    @property
    def depth(self):
        return self._a.depth + self._b.depth

    def _get_is_affine(self):
        return self._a.is_affine and self._b.is_affine
    is_affine = property(_get_is_affine)

    def _get_is_separable(self):
        return self._a.is_separable and self._b.is_separable
    is_separable = property(_get_is_separable)

    if DEBUG:
        def __str__(self):
            return '(%s, %s)' % (self._a, self._b)

    def __repr__(self):
        return "CompositeGenericTransform(%r, %r)" % (self._a, self._b)

    def transform_affine(self, points):
        return self.get_affine().transform(points)
    transform_affine.__doc__ = Transform.transform_affine.__doc__

    def transform_non_affine(self, points):
        if self._a.is_affine and self._b.is_affine:
            return points
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_non_affine(points)
        else:
            return self._b.transform_non_affine(
                                self._a.transform(points))
    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def transform_path_non_affine(self, path):
        if self._a.is_affine and self._b.is_affine:
            return path
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_path_non_affine(path)
        else:
            return self._b.transform_path_non_affine(
                                    self._a.transform_path(path))
    transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__

    def get_affine(self):
        if not self._b.is_affine:
            return self._b.get_affine()
        else:
            return Affine2D(np.dot(self._b.get_affine().get_matrix(),
                                self._a.get_affine().get_matrix()))
    get_affine.__doc__ = Transform.get_affine.__doc__

    def inverted(self):
        return CompositeGenericTransform(self._b.inverted(), self._a.inverted())
    inverted.__doc__ = Transform.inverted.__doc__

    def _get_has_inverse(self):
        return self._a.has_inverse and self._b.has_inverse
    has_inverse = property(_get_has_inverse)


class CompositeAffine2D(Affine2DBase):
    """
    A composite transform formed by applying transform *a* then transform *b*.

    This version is an optimization that handles the case where both *a*
    and *b* are 2D affines.
    """
    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        Both *a* and *b* must be instances of :class:`Affine2DBase`.

        You will generally not call this constructor directly but use
        the :func:`composite_transform_factory` function instead,
        which can automatically choose the best kind of composite
        transform instance to create.
        """
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims
        assert a.is_affine
        assert b.is_affine

        Affine2DBase.__init__(self, **kwargs)
        self._a = a
        self._b = b
        self.set_children(a, b)
        self._mtx = None

    if DEBUG:
        def __str__(self):
            return '(%s, %s)' % (self._a, self._b)

    @property
    def depth(self):
        return self._a.depth + self._b.depth

    def _iter_break_from_left_to_right(self):
        for lh_compliment, rh_compliment in self._a._iter_break_from_left_to_right():
            yield lh_compliment, rh_compliment + self._b
        for lh_compliment, rh_compliment in self._b._iter_break_from_left_to_right():
            yield self._a + lh_compliment, rh_compliment

    def __repr__(self):
        return "CompositeAffine2D(%r, %r)" % (self._a, self._b)

    def get_matrix(self):
        if self._invalid:
            self._mtx = np.dot(
                self._b.get_matrix(),
                self._a.get_matrix())
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


def composite_transform_factory(a, b):
    """
    Create a new composite transform that is the result of applying
    transform a then transform b.

    Shortcut versions of the blended transform are provided for the
    case where both child transforms are affine, or one or the other
    is the identity transform.

    Composite transforms may also be created using the '+' operator,
    e.g.::

      c = a + b
    """
    # check to see if any of a or b are IdentityTransforms. We use
    # isinstance here to guarantee that the transforms will *always*
    # be IdentityTransforms. Since TransformWrappers are mutable,
    # use of equality here would be wrong.
    if isinstance(a, IdentityTransform):
        return b
    elif isinstance(b, IdentityTransform):
        return a
    elif isinstance(a, Affine2D) and isinstance(b, Affine2D):
        return CompositeAffine2D(a, b)
    return CompositeGenericTransform(a, b)


class BboxTransform(Affine2DBase):
    """
    :class:`BboxTransform` linearly transforms points from one
    :class:`Bbox` to another :class:`Bbox`.
    """
    is_separable = True

    def __init__(self, boxin, boxout, **kwargs):
        """
        Create a new :class:`BboxTransform` that linearly transforms
        points from *boxin* to *boxout*.
        """
        assert boxin.is_bbox
        assert boxout.is_bbox

        Affine2DBase.__init__(self, **kwargs)
        self._boxin = boxin
        self._boxout = boxout
        self.set_children(boxin, boxout)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransform(%r, %r)" % (self._boxin, self._boxout)

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            outl, outb, outw, outh = self._boxout.bounds
            x_scale = outw / inw
            y_scale = outh / inh
            if DEBUG and (x_scale == 0 or y_scale == 0):
                raise ValueError("Transforming from or to a singular bounding box.")
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale+outl)],
                                   [0.0    , y_scale, (-inb*y_scale+outb)],
                                   [0.0    , 0.0    , 1.0        ]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformTo(Affine2DBase):
    """
    :class:`BboxTransformTo` is a transformation that linearly
    transforms points from the unit bounding box to a given
    :class:`Bbox`.
    """
    is_separable = True

    def __init__(self, boxout, **kwargs):
        """
        Create a new :class:`BboxTransformTo` that linearly transforms
        points from the unit bounding box to *boxout*.
        """
        assert boxout.is_bbox

        Affine2DBase.__init__(self, **kwargs)
        self._boxout = boxout
        self.set_children(boxout)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransformTo(%r)" % (self._boxout)

    def get_matrix(self):
        if self._invalid:
            outl, outb, outw, outh = self._boxout.bounds
            if DEBUG and (outw == 0 or outh == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[outw,  0.0, outl],
                                   [ 0.0, outh, outb],
                                   [ 0.0,  0.0,  1.0]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformToMaxOnly(BboxTransformTo):
    """
    :class:`BboxTransformTo` is a transformation that linearly
    transforms points from the unit bounding box to a given
    :class:`Bbox` with a fixed upper left of (0, 0).
    """
    def __repr__(self):
        return "BboxTransformToMaxOnly(%r)" % (self._boxout)

    def get_matrix(self):
        if self._invalid:
            xmax, ymax = self._boxout.max
            if DEBUG and (xmax == 0 or ymax == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[xmax,  0.0, 0.0],
                                  [ 0.0, ymax, 0.0],
                                  [ 0.0,  0.0, 1.0]],
                                 np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class BboxTransformFrom(Affine2DBase):
    """
    :class:`BboxTransformFrom` linearly transforms points from a given
    :class:`Bbox` to the unit bounding box.
    """
    is_separable = True

    def __init__(self, boxin, **kwargs):
        assert boxin.is_bbox

        Affine2DBase.__init__(self, **kwargs)
        self._boxin = boxin
        self.set_children(boxin)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransformFrom(%r)" % (self._boxin)

    def get_matrix(self):
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            if DEBUG and (inw == 0 or inh == 0):
                raise ValueError("Transforming from a singular bounding box.")
            x_scale = 1.0 / inw
            y_scale = 1.0 / inh
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale)],
                                   [0.0    , y_scale, (-inb*y_scale)],
                                   [0.0    , 0.0    , 1.0        ]],
                                  np.float_)
            self._inverted = None
            self._invalid = 0
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class ScaledTranslation(Affine2DBase):
    """
    A transformation that translates by *xt* and *yt*, after *xt* and *yt*
    have been transformad by the given transform *scale_trans*.
    """
    def __init__(self, xt, yt, scale_trans, **kwargs):
        Affine2DBase.__init__(self, **kwargs)
        self._t = (xt, yt)
        self._scale_trans = scale_trans
        self.set_children(scale_trans)
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "ScaledTranslation(%r)" % (self._t,)

    def get_matrix(self):
        if self._invalid:
            xt, yt = self._scale_trans.transform_point(self._t)
            self._mtx = np.array([[1.0, 0.0, xt],
                                   [0.0, 1.0, yt],
                                   [0.0, 0.0, 1.0]],
                                  np.float_)
            self._invalid = 0
            self._inverted = None
        return self._mtx
    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class TransformedPath(TransformNode):
    """
    A :class:`TransformedPath` caches a non-affine transformed copy of
    the :class:`~matplotlib.path.Path`.  This cached copy is
    automatically updated when the non-affine part of the transform
    changes.

    .. note::

        Paths are considered immutable by this class. Any update to the
        path's vertices/codes will not trigger a transform recomputation.

    """
    def __init__(self, path, transform):
        """
        Create a new :class:`TransformedPath` from the given
        :class:`~matplotlib.path.Path` and :class:`Transform`.
        """
        assert isinstance(transform, Transform)
        TransformNode.__init__(self)

        self._path = path
        self._transform = transform
        self.set_children(transform)
        self._transformed_path = None
        self._transformed_points = None

    def _revalidate(self):
        # only recompute if the invalidation includes the non_affine part of the transform
        if ((self._invalid & self.INVALID_NON_AFFINE == self.INVALID_NON_AFFINE)
            or self._transformed_path is None):
            self._transformed_path = \
                self._transform.transform_path_non_affine(self._path)
            self._transformed_points = \
                Path(self._transform.transform_non_affine(self._path.vertices),
                     None, self._path._interpolation_steps)
        self._invalid = 0

    def get_transformed_points_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.  Unlike
        :meth:`get_transformed_path_and_affine`, no interpolation will
        be performed.
        """
        self._revalidate()
        return self._transformed_points, self.get_affine()

    def get_transformed_path_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.
        """
        self._revalidate()
        return self._transformed_path, self.get_affine()

    def get_fully_transformed_path(self):
        """
        Return a fully-transformed copy of the child path.
        """
        self._revalidate()
        return self._transform.transform_path_affine(self._transformed_path)

    def get_affine(self):
        return self._transform.get_affine()


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    '''
    Modify the endpoints of a range as needed to avoid singularities.

    *vmin*, *vmax*
        the initial endpoints.

    *tiny*
        threshold for the ratio of the interval to the maximum absolute
        value of its endpoints.  If the interval is smaller than
        this, it will be expanded.  This value should be around
        1e-15 or larger; otherwise the interval will be approaching
        the double precision resolution limit.

    *expander*
        fractional amount by which *vmin* and *vmax* are expanded if
        the original interval is too small, based on *tiny*.

    *increasing*: [True | False]
        If True (default), swap *vmin*, *vmax* if *vmin* > *vmax*

    Returns *vmin*, *vmax*, expanded and/or swapped if necessary.

    If either input is inf or NaN, or if both inputs are 0,
    returns -*expander*, *expander*.
    '''
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return -expander, expander
    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True
    if vmax - vmin <= max(abs(vmin), abs(vmax)) * tiny:
        if vmax == 0 and vmin == 0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)

    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def interval_contains(interval, val):
    a, b = interval
    return (
        ((a < b) and (a <= val and b >= val))
        or (b <= val and a >= val))

def interval_contains_open(interval, val):
    a, b = interval
    return (
        ((a < b) and (a < val and b > val))
        or (b < val and a > val))

def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    '''
    Return a new transform with an added offset.
      args:
        trans is any transform
      kwargs:
        fig is the current figure; it can be None if units are 'dots'
        x, y give the offset
        units is 'inches', 'points' or 'dots'
    '''
    if units == 'dots':
        return trans + Affine2D().translate(x, y)
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')
    if units == 'points':
        x /= 72.0
        y /= 72.0
    elif not units == 'inches':
        raise ValueError('units must be dots, points, or inches')
    return trans + ScaledTranslation(x, y, fig.dpi_scale_trans)
