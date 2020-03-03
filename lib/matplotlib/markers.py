r"""
This module contains functions to handle markers.  Used by both the
marker functionality of `~matplotlib.axes.Axes.plot` and
`~matplotlib.axes.Axes.scatter`.

All possible markers are defined here:

============================== ====== =========================================
marker                         symbol description
============================== ====== =========================================
``"."``                        |m00|  point
``","``                        |m01|  pixel
``"o"``                        |m02|  circle
``"v"``                        |m03|  triangle_down
``"^"``                        |m04|  triangle_up
``"<"``                        |m05|  triangle_left
``">"``                        |m06|  triangle_right
``"1"``                        |m07|  tri_down
``"2"``                        |m08|  tri_up
``"3"``                        |m09|  tri_left
``"4"``                        |m10|  tri_right
``"8"``                        |m11|  octagon
``"s"``                        |m12|  square
``"p"``                        |m13|  pentagon
``"P"``                        |m23|  plus (filled)
``"*"``                        |m14|  star
``"h"``                        |m15|  hexagon1
``"H"``                        |m16|  hexagon2
``"+"``                        |m17|  plus
``"x"``                        |m18|  x
``"X"``                        |m24|  x (filled)
``"D"``                        |m19|  diamond
``"d"``                        |m20|  thin_diamond
``"|"``                        |m21|  vline
``"_"``                        |m22|  hline
``0`` (``TICKLEFT``)           |m25|  tickleft
``1`` (``TICKRIGHT``)          |m26|  tickright
``2`` (``TICKUP``)             |m27|  tickup
``3`` (``TICKDOWN``)           |m28|  tickdown
``4`` (``CARETLEFT``)          |m29|  caretleft
``5`` (``CARETRIGHT``)         |m30|  caretright
``6`` (``CARETUP``)            |m31|  caretup
``7`` (``CARETDOWN``)          |m32|  caretdown
``8`` (``CARETLEFTBASE``)      |m33|  caretleft (centered at base)
``9`` (``CARETRIGHTBASE``)     |m34|  caretright (centered at base)
``10`` (``CARETUPBASE``)       |m35|  caretup (centered at base)
``11`` (``CARETDOWNBASE``)     |m36|  caretdown (centered at base)
``"None"``, ``" "`` or  ``""``        nothing
``'$...$'``                    |m37|  Render the string using mathtext.
                                      E.g ``"$f$"`` for marker showing the
                                      letter ``f``.
``verts``                             A list of (x, y) pairs used for Path
                                      vertices. The center of the marker is
                                      located at (0, 0) and the size is
                                      normalized, such that the created path
                                      is encapsulated inside the unit cell.
path                                  A `~matplotlib.path.Path` instance.
``(numsides, 0, angle)``              A regular polygon with ``numsides``
                                      sides, rotated by ``angle``.
``(numsides, 1, angle)``              A star-like symbol with ``numsides``
                                      sides, rotated by ``angle``.
``(numsides, 2, angle)``              An asterisk with ``numsides`` sides,
                                      rotated by ``angle``.
============================== ====== =========================================

``None`` is the default which means 'nothing', however this table is
referred to from other docs for the valid inputs from marker inputs and in
those cases ``None`` still means 'default'.

Note that special symbols can be defined via the
:doc:`STIX math font </tutorials/text/mathtext>`,
e.g. ``"$\u266B$"``. For an overview over the STIX font symbols refer to the
`STIX font table <http://www.stixfonts.org/allGlyphs.html>`_.
Also see the :doc:`/gallery/text_labels_and_annotations/stix_fonts_demo`.

Integer numbers from ``0`` to ``11`` create lines and triangles. Those are
equally accessible via capitalized variables, like ``CARETDOWNBASE``.
Hence the following are equivalent::

    plt.plot([1, 2, 3], marker=11)
    plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)

Examples showing the use of markers:

* :doc:`/gallery/lines_bars_and_markers/marker_reference`
* :doc:`/gallery/shapes_and_collections/marker_path`
* :doc:`/gallery/lines_bars_and_markers/scatter_star_poly`


.. |m00| image:: /_static/markers/m00.png
.. |m01| image:: /_static/markers/m01.png
.. |m02| image:: /_static/markers/m02.png
.. |m03| image:: /_static/markers/m03.png
.. |m04| image:: /_static/markers/m04.png
.. |m05| image:: /_static/markers/m05.png
.. |m06| image:: /_static/markers/m06.png
.. |m07| image:: /_static/markers/m07.png
.. |m08| image:: /_static/markers/m08.png
.. |m09| image:: /_static/markers/m09.png
.. |m10| image:: /_static/markers/m10.png
.. |m11| image:: /_static/markers/m11.png
.. |m12| image:: /_static/markers/m12.png
.. |m13| image:: /_static/markers/m13.png
.. |m14| image:: /_static/markers/m14.png
.. |m15| image:: /_static/markers/m15.png
.. |m16| image:: /_static/markers/m16.png
.. |m17| image:: /_static/markers/m17.png
.. |m18| image:: /_static/markers/m18.png
.. |m19| image:: /_static/markers/m19.png
.. |m20| image:: /_static/markers/m20.png
.. |m21| image:: /_static/markers/m21.png
.. |m22| image:: /_static/markers/m22.png
.. |m23| image:: /_static/markers/m23.png
.. |m24| image:: /_static/markers/m24.png
.. |m25| image:: /_static/markers/m25.png
.. |m26| image:: /_static/markers/m26.png
.. |m27| image:: /_static/markers/m27.png
.. |m28| image:: /_static/markers/m28.png
.. |m29| image:: /_static/markers/m29.png
.. |m30| image:: /_static/markers/m30.png
.. |m31| image:: /_static/markers/m31.png
.. |m32| image:: /_static/markers/m32.png
.. |m33| image:: /_static/markers/m33.png
.. |m34| image:: /_static/markers/m34.png
.. |m35| image:: /_static/markers/m35.png
.. |m36| image:: /_static/markers/m36.png
.. |m37| image:: /_static/markers/m37.png
"""

from collections.abc import Sized
from collections import namedtuple

import numpy as np

from . import cbook, rcParams
from .path import Path
from .transforms import IdentityTransform, Affine2D, Bbox

# special-purpose marker identifiers:
(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN,
 CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
 CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE) = range(12)

_empty_path = Path(np.empty((0, 2)))

# hold info to track how marker size scales with increased "edge" thickness
PathEndAngle = namedtuple('PathEndAngle', 'incidence_angle corner_angle')
r"""Used to have a universal way to account for how much the bounding box of a
shape will grow as we increase its `markeredgewidth`.

Attributes
----------
    `incidence_angle` : float
        the angle that the corner bisector makes with the box edge (where
        top/bottom box edges are horizontal, left/right box edges are
        vertical).
    `corner_angle` : float
        the internal angle of the corner, where np.pi is a straight line, and 0
        is retracing exactly the way you came. None can be used to signify that
        the line ends there (i.e. no corner).

Notes
-----
$\pi$ and 0 are equivalent for `corner_angle`. Both $\theta$ and $\pi - \theta$
are equivalent for `incidence_angle` by symmetry."""

BoxSides = namedtuple('BoxSides', 'top bottom left right')
"""Easily keep track of same parameter for each of four sides."""

# some angles are heavily repeated throughout various markers
_tri_side_angle = np.arctan(2)
_tri_tip_angle = np.arctan(1/2)
# half the edge length of the smaller pentagon over the difference between the
# larger pentagon's circumcribing radius and the smaller pentagon's inscribed
# radius
_star_tip_angle = 2*np.arctan2((1/4)*np.sqrt((5 - np.sqrt(5))/2),
                               1 - np.sqrt((3 + np.sqrt(5))/32))
# reusable corner types
_flat_side = PathEndAngle(0, 0)
_normal_line = PathEndAngle(np.pi/2, None)
_normal_right_angle = PathEndAngle(np.pi/2, np.pi/2)
_tri_side_corner = PathEndAngle(np.pi/2 - _tri_side_angle/2, _tri_side_angle)
_tri_tip = PathEndAngle(np.pi/2, _tri_tip_angle)
# and some entire box side behaviors are repeated among markers
_effective_square = BoxSides(_flat_side, _flat_side, _flat_side, _flat_side)
_effective_diamond = BoxSides(_normal_right_angle, _normal_right_angle,
                              _normal_right_angle, _normal_right_angle)

# precomputed information required for marker_bbox (besides _joinstyle)
_edge_angles = {
    '.': _effective_square,
    ',': _effective_square,
    'o': _effective_square,
    # hit two corners and tip bisects one side of unit square
    'v': BoxSides(_flat_side, _tri_tip, _tri_side_corner, _tri_side_corner),
    '^': BoxSides(_tri_tip, _flat_side, _tri_side_corner, _tri_side_corner),
    '<': BoxSides(_tri_side_corner, _tri_side_corner, _tri_tip, _flat_side),
    '>': BoxSides(_tri_side_corner, _tri_side_corner, _flat_side, _tri_tip),
    # angle bisectors of an equilateral triangle. lines of length 1/2
    '1': BoxSides(PathEndAngle(np.pi/6, None), PathEndAngle(np.pi/2, None),
                  PathEndAngle(np.pi/3, None), PathEndAngle(np.pi/3, None)),
    '2': BoxSides(PathEndAngle(np.pi/2, None), PathEndAngle(np.pi/6, None),
                  PathEndAngle(np.pi/3, None), PathEndAngle(np.pi/3, None)),
    '3': BoxSides(PathEndAngle(np.pi/3, None), PathEndAngle(np.pi/3, None),
                  PathEndAngle(np.pi/2, None), PathEndAngle(np.pi/6, None)),
    '4': BoxSides(PathEndAngle(np.pi/3, None), PathEndAngle(np.pi/3, None),
                  PathEndAngle(np.pi/6, None), PathEndAngle(np.pi/2, None)),
    # regular polygons, circumscribed in circle of radius 1.
    '8': _effective_square,
    's': _effective_square,
    'p': BoxSides(PathEndAngle(np.pi/2, 3*np.pi/5), _flat_side,
                  PathEndAngle(2*np.pi/5, 3*np.pi/5),
                  PathEndAngle(2*np.pi/5, 3*np.pi/5)),
    # tips are corners of regular pentagon circuscribed in circle of radius 1.
    # so incidence angles are same as pentagon
    # interior points are corners of another regular pentagon, whose
    # circumscribing circle has radius 0.5, so all tip angles are same
    '*': BoxSides(PathEndAngle(np.pi/2, _star_tip_angle),
                  PathEndAngle(3*np.pi/10, _star_tip_angle),
                  PathEndAngle(2*np.pi/5, _star_tip_angle),
                  PathEndAngle(2*np.pi/5, _star_tip_angle)),
    'h': BoxSides(PathEndAngle(np.pi/2, 2*np.pi/3),
                  PathEndAngle(np.pi/2, 2*np.pi/3),
                  _flat_side, _flat_side),
    'H': BoxSides(_flat_side, _flat_side,
                  PathEndAngle(np.pi/2, 2*np.pi/3),
                  PathEndAngle(np.pi/2, 2*np.pi/3)),
    '+': BoxSides(_normal_line, _normal_line, _normal_line, _normal_line),
    'x': BoxSides(PathEndAngle(np.pi/4, None), PathEndAngle(np.pi/4, None),
                  PathEndAngle(np.pi/4, None), PathEndAngle(np.pi/4, None)),
    # unit square rotated pi/2
    'D': _effective_diamond,
    # D scaled by 0.6 in horizontal direction
    'd': BoxSides(PathEndAngle(np.pi/2, 2*np.arctan(3/5)),
                  PathEndAngle(np.pi/2, 2*np.arctan(3/5)),
                  PathEndAngle(np.pi/2, 2*np.arctan(5/3)),
                  PathEndAngle(np.pi/2, 2*np.arctan(5/3))),
    '|': BoxSides(_normal_line, _normal_line, _flat_side, _flat_side),
    '_': BoxSides(_flat_side, _flat_side, _normal_line, _normal_line),
    'P': _effective_square,
    'X': _effective_diamond,
    TICKLEFT: BoxSides(_flat_side, _flat_side, _normal_line, _normal_line),
    TICKRIGHT: BoxSides(_flat_side, _flat_side, _normal_line, _normal_line),
    TICKUP: BoxSides(_normal_line, _normal_line, _flat_side, _flat_side),
    TICKDOWN: BoxSides(_normal_line, _normal_line, _flat_side, _flat_side),
    # carets same size as "triangles" but missing the edge opposite their "tip"
    CARETLEFT: BoxSides(PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                        PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                        _tri_tip, PathEndAngle(_tri_side_angle, None)),
    CARETRIGHT: BoxSides(PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                         PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                         PathEndAngle(_tri_side_angle, None), _tri_tip),
    CARETUP: BoxSides(_tri_tip, PathEndAngle(_tri_side_angle, None),
                      PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                      PathEndAngle(np.pi/2 - _tri_side_angle/2, None)),
    CARETDOWN: BoxSides(PathEndAngle(_tri_side_angle, None), _tri_tip,
                        PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                        PathEndAngle(np.pi/2 - _tri_side_angle/2, None)),
    CARETLEFTBASE: BoxSides(PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                            PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                            _tri_tip, PathEndAngle(_tri_side_angle, None)),
    CARETRIGHTBASE: BoxSides(PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                             PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                             PathEndAngle(_tri_side_angle, None), _tri_tip),
    CARETUPBASE: BoxSides(_tri_tip, PathEndAngle(_tri_side_angle, None),
                          PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                          PathEndAngle(np.pi/2 - _tri_side_angle/2, None)),
    CARETDOWNBASE: BoxSides(PathEndAngle(_tri_side_angle, None), _tri_tip,
                            PathEndAngle(np.pi/2 - _tri_side_angle/2, None),
                            PathEndAngle(np.pi/2 - _tri_side_angle/2, None)),
    '': BoxSides(None, None, None, None),
    ' ': BoxSides(None, None, None, None),
    'None': BoxSides(None, None, None, None),
    None: BoxSides(None, None, None, None),
}


def _get_padding_due_to_angle(width, path_end_angle, joinstyle='miter',
                              capstyle='butt'):
    """How much does adding a stroke with `width` overflow the naive bbox at a
    corner described by `path_end_angle`?

    Parameters
    ----------
    width : float
        `markeredgewidth` used to draw the stroke that we're computing the
        overflow of
    path_end_angle : PathEndAngle
        precomputed property of a corner that allows us to compute the overflow

    Returns
    -------
    pad : float
        amount of overflow

    """
    if path_end_angle is None or width == 0:
        return 0
    phi, theta = path_end_angle.incidence_angle, path_end_angle.corner_angle
    # if there's no corner (i.e. the path just ends, as in the "sides" of the
    # carets or in the non-fillable markers, we can compute how far the outside
    # edge of the markeredge stroke extends outside of the bounding box of its
    # path using the law of sines: $\sin(\phi)/(w/2) = \sin(\pi/2 - \phi)/l$
    # for $w$ the `markeredgewidth`, $\phi$ the incidence angle of the line,
    # then $l$ is the length along the outer edge of the stroke that extends
    # beyond the bouding box. We can translate this to a distance perpendicular
    # to the bounding box E(w, \phi) = l \sin(\phi)$, for $l$ as above.
    if theta is None:
        # also note that in this case, we shouldn't check _joinstyle because
        # it's going to be "round" by default but not actually be in use. what
        # we care about is _capstyle, which is currently always its default
        # "butt" for all markers. if we find otherwise, we should change this
        # code to check for the "projecting" and "round" cases
        if capstyle != 'butt':
            raise NotImplementedError("Only capstyle='butt' currently needed")
        pad = (width/2) * np.cos(phi)
    # to calculate the offset for _joinstyle == 'miter', imagine aligning the
    # corner so that on line comes in along the negative x-axis, and another
    # from above, makes an angle $\theta$ with the negative x-axis.  the tip of
    # the new corner created by the markeredge stroke will be at the point
    # where the two outer edge of the markeredge stroke intersect.  in the
    # orientation described above, the outer edge of the stroke aligned with
    # the x axis will obviously have equation $y = -w/2$ where $w$ is the
    # markeredgewidth. WLOG, the stroke coming in from above at an angle
    # $\theta$ from the negative x-axis will have equation
    # $$-(\tan(\theta) x + \frac{w}{2\cos(\theta)}.$$
    # the intersection of these two lines is at $y = w/2$, and we can solve for
    # $x = \cot(\theta) (\frac{w}{2} + \frac{w}{2\cos(\theta)})$.
    # this puts the "edge" tip a distance $M = (w/2)\sqrt{\csc^2(\theta/2)+1}$
    # from the tip of the corner itself, on the line defined by the bisector of
    # the corner angle. So the extra padding required is $M\sin(\phi)$, where
    # $\phi$ is the incidence angle of the corner's bisector
    elif joinstyle == 'miter':
        pad = (width/2)*np.sin(phi)*np.sqrt(np.power(np.sin(theta/2), -2) + 1)
    # to calculate the offset for _joinstyle = "bevel", we can start with the
    # analogous "miter" corner. the rules for how the "bevel" is
    # created in SVG is that the outer edges of the stroke continue up until
    # the stroke hits the corner point (similar to _capstyle='butt'). A line is
    # then drawn joining these two outer points and the interior is filled. in
    # other words, it is the same as a "miter" corner, but with some amount of
    # the tip removed (an isoceles triangle with base given by the distance
    # described above). This base length (the bevel "size") is given by the law
    # of sines $b = (w/2)\frac{\sin(\pi - \theta)}{\sin(\theta/2)}$.
    # We can then subtract the height of the isoceles rectangle with this base
    # height and tip angle $\theta$ from our result $M$ above to get how far
    # the midpoint of the bevel extends beyond the outside....

    # but that's not what we're interested in.
    # a beveled edge is exactly the convex hull of its two composite lines with
    # capstyle='butt'. So we just compute the individual lines' incidence
    # angles and take the maximum of the two padding values
    elif joinstyle == 'bevel':
        phi1 = phi + theta/2
        phi2 = phi - theta/2
        pad = (width/2) * max(np.cos(phi1), np.cos(phi2))
    # finally, _joinstyle = "round" is just _joinstyle = "bevel" but with
    # a hemispherical cap. we could calculate this but for now no markers use
    # it....except those with "no corner", in which case we can treat them the
    # same as squares...
    elif joinstyle == 'round':
        # the two "same as straight line" cases
        if np.isclose(theta, 0) and np.isclose(phi, 0) \
        or np.isclose(theta, np.pi) and np.isclose(phi, np.pi/2):
            pad = width/2
        else:
            raise NotImplementedError("Only 'miter' and 'bevel' joinstyles "
                                      "needed for now")
    else:
        raise ValueError(f"Unknown joinstyle: {joinstyle}")
    return pad


class MarkerStyle:

    markers = {
        '.': 'point',
        ',': 'pixel',
        'o': 'circle',
        'v': 'triangle_down',
        '^': 'triangle_up',
        '<': 'triangle_left',
        '>': 'triangle_right',
        '1': 'tri_down',
        '2': 'tri_up',
        '3': 'tri_left',
        '4': 'tri_right',
        '8': 'octagon',
        's': 'square',
        'p': 'pentagon',
        '*': 'star',
        'h': 'hexagon1',
        'H': 'hexagon2',
        '+': 'plus',
        'x': 'x',
        'D': 'diamond',
        'd': 'thin_diamond',
        '|': 'vline',
        '_': 'hline',
        'P': 'plus_filled',
        'X': 'x_filled',
        TICKLEFT: 'tickleft',
        TICKRIGHT: 'tickright',
        TICKUP: 'tickup',
        TICKDOWN: 'tickdown',
        CARETLEFT: 'caretleft',
        CARETRIGHT: 'caretright',
        CARETUP: 'caretup',
        CARETDOWN: 'caretdown',
        CARETLEFTBASE: 'caretleftbase',
        CARETRIGHTBASE: 'caretrightbase',
        CARETUPBASE: 'caretupbase',
        CARETDOWNBASE: 'caretdownbase',
        "None": 'nothing',
        None: 'nothing',
        ' ': 'nothing',
        '': 'nothing'
    }

    # Just used for informational purposes.  is_filled()
    # is calculated in the _set_* functions.
    filled_markers = (
        'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
        'P', 'X')

    fillstyles = ('full', 'left', 'right', 'bottom', 'top', 'none')
    _half_fillstyles = ('left', 'right', 'bottom', 'top')

    def __init__(self, marker=None, fillstyle=None):
        """
        Attributes
        ----------
        markers : list of known marks

        fillstyles : list of known fillstyles

        filled_markers : list of known filled markers.

        Parameters
        ----------
        marker : str or array-like, optional, default: None
            See the descriptions of possible markers in the module docstring.

        fillstyle : str, optional, default: 'full'
            'full', 'left", 'right', 'bottom', 'top', 'none'
        """
        self._marker_function = None
        self.set_fillstyle(fillstyle)
        self.set_marker(marker)

    def _recache(self):
        if self._marker_function is None:
            return
        self._path = _empty_path
        self._transform = IdentityTransform()
        self._alt_path = None
        self._alt_transform = None
        self._snap_threshold = None
        self._joinstyle = 'round'
        self._capstyle = 'butt'
        self._filled = True
        self._marker_function()

    def __bool__(self):
        return bool(len(self._path.vertices))

    def is_filled(self):
        return self._filled

    def get_fillstyle(self):
        return self._fillstyle

    def set_fillstyle(self, fillstyle):
        """
        Sets fillstyle

        Parameters
        ----------
        fillstyle : string amongst known fillstyles
        """
        if fillstyle is None:
            fillstyle = rcParams['markers.fillstyle']
        cbook._check_in_list(self.fillstyles, fillstyle=fillstyle)
        self._fillstyle = fillstyle
        self._recache()

    def get_joinstyle(self):
        return self._joinstyle

    def get_capstyle(self):
        return self._capstyle

    def get_marker(self):
        return self._marker

    def set_marker(self, marker):
        if (isinstance(marker, np.ndarray) and marker.ndim == 2 and
                marker.shape[1] == 2):
            self._marker_function = self._set_vertices
        elif isinstance(marker, str) and cbook.is_math_text(marker):
            self._marker_function = self._set_mathtext_path
        elif isinstance(marker, Path):
            self._marker_function = self._set_path_marker
        elif (isinstance(marker, Sized) and len(marker) in (2, 3) and
                marker[1] in (0, 1, 2)):
            self._marker_function = self._set_tuple_marker
        elif (not isinstance(marker, (np.ndarray, list)) and
              marker in self.markers):
            self._marker_function = getattr(
                self, '_set_' + self.markers[marker])
        else:
            try:
                Path(marker)
                self._marker_function = self._set_vertices
            except ValueError:
                raise ValueError('Unrecognized marker style {!r}'
                                 .format(marker))

        self._marker = marker
        self._recache()

    def get_path(self):
        return self._path

    def get_transform(self):
        return self._transform.frozen()

    def get_alt_path(self):
        return self._alt_path

    def get_alt_transform(self):
        return self._alt_transform.frozen()

    def get_snap_threshold(self):
        return self._snap_threshold

    def _set_nothing(self):
        self._filled = False

    def _set_custom_marker(self, path):
        rescale = np.max(np.abs(path.vertices))  # max of x's and y's.
        self._transform = Affine2D().scale(0.5 / rescale)
        self._path = path

    def _set_path_marker(self):
        self._set_custom_marker(self._marker)

    def _set_vertices(self):
        self._set_custom_marker(Path(self._marker))

    def _set_tuple_marker(self):
        marker = self._marker
        if len(marker) == 2:
            numsides, rotation = marker[0], 0.0
        elif len(marker) == 3:
            numsides, rotation = marker[0], marker[2]
        symstyle = marker[1]
        if symstyle == 0:
            self._path = Path.unit_regular_polygon(numsides)
            self._joinstyle = 'miter'
        elif symstyle == 1:
            self._path = Path.unit_regular_star(numsides)
            self._joinstyle = 'bevel'
        elif symstyle == 2:
            self._path = Path.unit_regular_asterisk(numsides)
            self._filled = False
            self._joinstyle = 'bevel'
        else:
            raise ValueError(f"Unexpected tuple marker: {marker}")

    def _set_mathtext_path(self):
        """
        Draws mathtext markers '$...$' using TextPath object.

        Submitted by tcb
        """
        from matplotlib.text import TextPath

        # again, the properties could be initialised just once outside
        # this function
        text = TextPath(xy=(0, 0), s=self.get_marker(),
                        usetex=rcParams['text.usetex'])
        if len(text.vertices) == 0:
            return

        xmin, ymin = text.vertices.min(axis=0)
        xmax, ymax = text.vertices.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin
        max_dim = max(width, height)
        self._transform = Affine2D() \
            .translate(-xmin + 0.5 * -width, -ymin + 0.5 * -height) \
            .scale(1.0 / max_dim)
        self._path = text
        self._snap = False

    def _half_fill(self):
        return self.get_fillstyle() in self._half_fillstyles

    def _set_circle(self, reduction=1.0):
        self._transform = Affine2D().scale(0.5 * reduction)
        self._snap_threshold = np.inf
        fs = self.get_fillstyle()
        if not self._half_fill():
            self._path = Path.unit_circle()
        else:
            # build a right-half circle
            if fs == 'bottom':
                rotate = 270.
            elif fs == 'top':
                rotate = 90.
            elif fs == 'left':
                rotate = 180.
            else:
                rotate = 0.

            self._path = self._alt_path = Path.unit_circle_righthalf()
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform.frozen().rotate_deg(180.)

    def _set_pixel(self):
        self._path = Path.unit_rectangle()
        # Ideally, you'd want -0.5, -0.5 here, but then the snapping
        # algorithm in the Agg backend will round this to a 2x2
        # rectangle from (-1, -1) to (1, 1).  By offsetting it
        # slightly, we can force it to be (0, 0) to (1, 1), which both
        # makes it only be a single pixel and places it correctly
        # aligned to 1-width stroking (i.e. the ticks).  This hack is
        # the best of a number of bad alternatives, mainly because the
        # backends are not aware of what marker is actually being used
        # beyond just its path data.
        self._transform = Affine2D().translate(-0.49999, -0.49999)
        self._snap_threshold = None

    def _set_point(self):
        # a "point" is defined to a circle with half the requested markersize
        self._set_circle(reduction=0.5)

    _triangle_path = Path(
        [[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # Going down halfway looks to small.  Golden ratio is too far.
    _triangle_path_u = Path(
        [[0.0, 1.0], [-3 / 5., -1 / 5.], [3 / 5., -1 / 5.], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_d = Path(
        [[-3 / 5., -1 / 5.], [3 / 5., -1 / 5.], [1.0, -1.0], [-1.0, -1.0],
         [-3 / 5., -1 / 5.]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_l = Path(
        [[0.0, 1.0], [0.0, -1.0], [-1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    _triangle_path_r = Path(
        [[0.0, 1.0], [0.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    def _set_triangle(self, rot, skip):
        self._transform = Affine2D().scale(0.5).rotate_deg(rot)
        self._snap_threshold = 5.0
        fs = self.get_fillstyle()

        if not self._half_fill():
            self._path = self._triangle_path
        else:
            mpaths = [self._triangle_path_u,
                      self._triangle_path_l,
                      self._triangle_path_d,
                      self._triangle_path_r]

            if fs == 'top':
                self._path = mpaths[(0 + skip) % 4]
                self._alt_path = mpaths[(2 + skip) % 4]
            elif fs == 'bottom':
                self._path = mpaths[(2 + skip) % 4]
                self._alt_path = mpaths[(0 + skip) % 4]
            elif fs == 'left':
                self._path = mpaths[(1 + skip) % 4]
                self._alt_path = mpaths[(3 + skip) % 4]
            else:
                self._path = mpaths[(3 + skip) % 4]
                self._alt_path = mpaths[(1 + skip) % 4]

            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_triangle_up(self):
        return self._set_triangle(0.0, 0)

    def _set_triangle_down(self):
        return self._set_triangle(180.0, 2)

    def _set_triangle_left(self):
        return self._set_triangle(90.0, 3)

    def _set_triangle_right(self):
        return self._set_triangle(270.0, 1)

    def _set_square(self):
        self._transform = Affine2D().translate(-0.5, -0.5)
        self._snap_threshold = 2.0
        fs = self.get_fillstyle()
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            # build a bottom filled square out of two rectangles, one
            # filled.  Use the rotation to support left, right, bottom
            # or top
            if fs == 'bottom':
                rotate = 0.
            elif fs == 'top':
                rotate = 180.
            elif fs == 'left':
                rotate = 270.
            else:
                rotate = 90.

            self._path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
                               [0.0, 0.5], [0.0, 0.0]])
            self._alt_path = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0],
                                   [0.0, 1.0], [0.0, 0.5]])
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_diamond(self):
        self._transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
        self._snap_threshold = 5.0
        fs = self.get_fillstyle()
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            self._path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
            self._alt_path = Path([[0.0, 0.0], [0.0, 1.0],
                                   [1.0, 1.0], [0.0, 0.0]])

            if fs == 'bottom':
                rotate = 270.
            elif fs == 'top':
                rotate = 90.
            elif fs == 'left':
                rotate = 180.
            else:
                rotate = 0.

            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_thin_diamond(self):
        self._set_diamond()
        self._transform.scale(0.6, 1.0)

    def _set_pentagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        polypath = Path.unit_regular_polygon(5)
        fs = self.get_fillstyle()

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices

            y = (1 + np.sqrt(5)) / 4.
            top = Path([verts[0], verts[1], verts[4], verts[0]])
            bottom = Path([verts[1], verts[2], verts[3], verts[4], verts[1]])
            left = Path([verts[0], verts[1], verts[2], [0, -y], verts[0]])
            right = Path([verts[0], verts[4], verts[3], [0, -y], verts[0]])

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left
            self._path = mpath
            self._alt_path = mpath_alt
            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_star(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        fs = self.get_fillstyle()
        polypath = Path.unit_regular_star(5, innerCircle=0.381966)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices

            top = Path(np.vstack((verts[0:4, :], verts[7:10, :], verts[0])))
            bottom = Path(np.vstack((verts[3:8, :], verts[3])))
            left = Path(np.vstack((verts[0:6, :], verts[0])))
            right = Path(np.vstack((verts[0], verts[5:10, :], verts[0])))

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left
            self._path = mpath
            self._alt_path = mpath_alt
            self._alt_transform = self._transform

        self._joinstyle = 'bevel'

    def _set_hexagon1(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = None

        fs = self.get_fillstyle()
        polypath = Path.unit_regular_polygon(6)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices

            # not drawing inside lines
            x = np.abs(np.cos(5 * np.pi / 6.))
            top = Path(np.vstack(([-x, 0], verts[(1, 0, 5), :], [x, 0])))
            bottom = Path(np.vstack(([-x, 0], verts[2:5, :], [x, 0])))
            left = Path(verts[(0, 1, 2, 3), :])
            right = Path(verts[(0, 5, 4, 3), :])

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            self._path = mpath
            self._alt_path = mpath_alt
            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_hexagon2(self):
        self._transform = Affine2D().scale(0.5).rotate_deg(30)
        self._snap_threshold = None

        fs = self.get_fillstyle()
        polypath = Path.unit_regular_polygon(6)

        if not self._half_fill():
            self._path = polypath
        else:
            verts = polypath.vertices

            # not drawing inside lines
            x, y = np.sqrt(3) / 4, 3 / 4.
            top = Path(verts[(1, 0, 5, 4, 1), :])
            bottom = Path(verts[(1, 2, 3, 4), :])
            left = Path(np.vstack(([x, y], verts[(0, 1, 2), :],
                                   [-x, -y], [x, y])))
            right = Path(np.vstack(([x, y], verts[(5, 4, 3), :], [-x, -y])))

            if fs == 'top':
                mpath, mpath_alt = top, bottom
            elif fs == 'bottom':
                mpath, mpath_alt = bottom, top
            elif fs == 'left':
                mpath, mpath_alt = left, right
            else:
                mpath, mpath_alt = right, left

            self._path = mpath
            self._alt_path = mpath_alt
            self._alt_transform = self._transform

        self._joinstyle = 'miter'

    def _set_octagon(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0

        fs = self.get_fillstyle()
        polypath = Path.unit_regular_polygon(8)

        if not self._half_fill():
            self._transform.rotate_deg(22.5)
            self._path = polypath
        else:
            x = np.sqrt(2.) / 4.
            half = Path([[0, -1], [0, 1], [-x, 1], [-1, x],
                         [-1, -x], [-x, -1], [0, -1]])

            if fs == 'bottom':
                rotate = 90.
            elif fs == 'top':
                rotate = 270.
            elif fs == 'right':
                rotate = 180.
            else:
                rotate = 0.

            self._transform.rotate_deg(rotate)
            self._path = self._alt_path = half
            self._alt_transform = self._transform.frozen().rotate_deg(180.0)

        self._joinstyle = 'miter'

    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])

    def _set_vline(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._line_marker_path

    def _set_hline(self):
        self._set_vline()
        self._transform = self._transform.rotate_deg(90)

    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])

    def _set_tickleft(self):
        self._transform = Affine2D().scale(-1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path

    def _set_tickright(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickhoriz_path

    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])

    def _set_tickup(self):
        self._transform = Affine2D().scale(1.0, 1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path

    def _set_tickdown(self):
        self._transform = Affine2D().scale(1.0, -1.0)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._tickvert_path

    _tri_path = Path([[0.0, 0.0], [0.0, -1.0],
                      [0.0, 0.0], [0.8, 0.5],
                      [0.0, 0.0], [-0.8, 0.5]],
                     [Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO])

    def _set_tri_down(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 5.0
        self._filled = False
        self._path = self._tri_path

    def _set_tri_up(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(180)

    def _set_tri_left(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(270)

    def _set_tri_right(self):
        self._set_tri_down()
        self._transform = self._transform.rotate_deg(90)

    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])

    def _set_caretdown(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._caret_path
        self._joinstyle = 'miter'

    def _set_caretup(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleft(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretright(self):
        self._set_caretdown()
        self._transform = self._transform.rotate_deg(90)

    _caret_path_base = Path([[-1.0, 0.0], [0.0, -1.5], [1.0, 0]])

    def _set_caretdownbase(self):
        self._set_caretdown()
        self._path = self._caret_path_base

    def _set_caretupbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(180)

    def _set_caretleftbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(270)

    def _set_caretrightbase(self):
        self._set_caretdownbase()
        self._transform = self._transform.rotate_deg(90)

    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0],
                       [0.0, -1.0], [0.0, 1.0]],
                      [Path.MOVETO, Path.LINETO,
                       Path.MOVETO, Path.LINETO])

    def _set_plus(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 1.0
        self._filled = False
        self._path = self._plus_path

    _x_path = Path([[-1.0, -1.0], [1.0, 1.0],
                    [-1.0, 1.0], [1.0, -1.0]],
                   [Path.MOVETO, Path.LINETO,
                    Path.MOVETO, Path.LINETO])

    def _set_x(self):
        self._transform = Affine2D().scale(0.5)
        self._snap_threshold = 3.0
        self._filled = False
        self._path = self._x_path

    _plus_filled_path = Path([(1/3, 0), (2/3, 0), (2/3, 1/3),
                              (1, 1/3), (1, 2/3), (2/3, 2/3),
                              (2/3, 1), (1/3, 1), (1/3, 2/3),
                              (0, 2/3), (0, 1/3), (1/3, 1/3),
                              (1/3, 0)],
                             [Path.MOVETO, Path.LINETO, Path.LINETO,
                              Path.LINETO, Path.LINETO, Path.LINETO,
                              Path.LINETO, Path.LINETO, Path.LINETO,
                              Path.LINETO, Path.LINETO, Path.LINETO,
                              Path.CLOSEPOLY])

    _plus_filled_path_t = Path([(1, 1/2), (1, 2/3), (2/3, 2/3),
                                (2/3, 1), (1/3, 1), (1/3, 2/3),
                                (0, 2/3), (0, 1/2), (1, 1/2)],
                               [Path.MOVETO, Path.LINETO, Path.LINETO,
                                Path.LINETO, Path.LINETO, Path.LINETO,
                                Path.LINETO, Path.LINETO,
                                Path.CLOSEPOLY])

    def _set_plus_filled(self):
        self._transform = Affine2D().translate(-0.5, -0.5)
        self._snap_threshold = 5.0
        self._joinstyle = 'miter'
        fs = self.get_fillstyle()
        if not self._half_fill():
            self._path = self._plus_filled_path
        else:
            # Rotate top half path to support all partitions
            if fs == 'top':
                rotate, rotate_alt = 0, 180
            elif fs == 'bottom':
                rotate, rotate_alt = 180, 0
            elif fs == 'left':
                rotate, rotate_alt = 90, 270
            else:
                rotate, rotate_alt = 270, 90

            self._path = self._plus_filled_path_t
            self._alt_path = self._plus_filled_path_t
            self._alt_transform = Affine2D().translate(-0.5, -0.5)
            self._transform.rotate_deg(rotate)
            self._alt_transform.rotate_deg(rotate_alt)

    _x_filled_path = Path([(0.25, 0), (0.5, 0.25), (0.75, 0), (1, 0.25),
                           (0.75, 0.5), (1, 0.75), (0.75, 1), (0.5, 0.75),
                           (0.25, 1), (0, 0.75), (0.25, 0.5), (0, 0.25),
                           (0.25, 0)],
                          [Path.MOVETO, Path.LINETO, Path.LINETO,
                           Path.LINETO, Path.LINETO, Path.LINETO,
                           Path.LINETO, Path.LINETO, Path.LINETO,
                           Path.LINETO, Path.LINETO, Path.LINETO,
                           Path.CLOSEPOLY])

    _x_filled_path_t = Path([(0.75, 0.5), (1, 0.75), (0.75, 1),
                             (0.5, 0.75), (0.25, 1), (0, 0.75),
                             (0.25, 0.5), (0.75, 0.5)],
                            [Path.MOVETO, Path.LINETO, Path.LINETO,
                             Path.LINETO, Path.LINETO, Path.LINETO,
                             Path.LINETO, Path.CLOSEPOLY])

    def _set_x_filled(self):
        self._transform = Affine2D().translate(-0.5, -0.5)
        self._snap_threshold = 5.0
        self._joinstyle = 'miter'
        fs = self.get_fillstyle()
        if not self._half_fill():
            self._path = self._x_filled_path
        else:
            # Rotate top half path to support all partitions
            if fs == 'top':
                rotate, rotate_alt = 0, 180
            elif fs == 'bottom':
                rotate, rotate_alt = 180, 0
            elif fs == 'left':
                rotate, rotate_alt = 90, 270
            else:
                rotate, rotate_alt = 270, 90

            self._path = self._x_filled_path_t
            self._alt_path = self._x_filled_path_t
            self._alt_transform = Affine2D().translate(-0.5, -0.5)
            self._transform.rotate_deg(rotate)
            self._alt_transform.rotate_deg(rotate_alt)


    def get_centered_bbox(self, markerwidth=0, markeredgewidth=0):
        """Get size of bbox if marker is centered at origin.

        For markers with no edge, this is just the same bbox as that of the
        transformed marker path, but how much extra extent is added by an edge
        is a function of the angle of the path at its own (the path's own)
        boundary.

        Parameters
        ----------
        markerwidth : float, optional, default: None
            "Size" of the marker, in pixels.

        markeredgewidth : float, optional, default: None
            Width, in pixels, of the stroke used to create the marker's edge.

        Returns
        -------

        bbox : matplotlib.transforms.Bbox
            The extents of the marker including its edge (in pixels) if it were
            centered at (0,0).

        Notes
        -----
        """
        if type(marker) != MarkerStyle:
            marker = MarkerStyle(marker)
        unit_path = self._transform.transform_path(self._path)
        unit_bbox = unit_path.get_extents()
        scale = Affine2D().scale(markerwidth)
        [[left, bottom], [right, top]] = scale.transform(unit_bbox)
        angles = _edge_angles[marker._marker]
        left -= _get_padding_due_to_angle(markeredgewidth, angles.left,
                                          self._joinstyle, self._capstyle)
        bottom -= _get_padding_due_to_angle(markeredgewidth, angles.bottom,
                                            self._joinstyle, self._capstyle)
        right += _get_padding_due_to_angle(markeredgewidth, angles.right,
                                           self._joinstyle, self._capstyle)
        top += _get_padding_due_to_angle(markeredgewidth, angles.top,
                                         self._joinstyle, self._capstyle)
        return Bbox.from_extents(left, bottom, right, top)


