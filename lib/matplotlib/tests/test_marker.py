import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison

from collections import namedtuple
import pytest


def test_markers_valid():
    marker_style = markers.MarkerStyle()
    mrk_array = np.array([[-0.5, 0],
                          [0.5, 0]])
    # Checking this doesn't fail.
    marker_style.set_marker(mrk_array)


def test_markers_invalid():
    marker_style = markers.MarkerStyle()
    mrk_array = np.array([[-0.5, 0, 1, 2, 3]])
    # Checking this does fail.
    with pytest.raises(ValueError):
        marker_style.set_marker(mrk_array)


def test_marker_path():
    marker_style = markers.MarkerStyle()
    path = Path([[0, 0], [1, 0]], [Path.MOVETO, Path.LINETO])
    # Checking this doesn't fail.
    marker_style.set_marker(path)


def _draw_marker_outlined(marker, markeredgewidth=0, markersize=100):
    # keep marker vaguely inside the figure by scaling
    fig_d = 4*(markeredgewidth + markersize)/100
    fig, ax = plt.subplots(figsize=(fig_d, fig_d))
    ax.axis('off')
    # and fix limits so pix size doesn't change later
    ax_lim = 2
    plt.xlim([-ax_lim, ax_lim])
    plt.ylim([-ax_lim, ax_lim])
    # draw a single marker centered at the origin
    lines = ax.plot(0, 0, 'b', markersize=markersize, marker=marker,
                    clip_on=False, markeredgewidth=markeredgewidth,
                    markeredgecolor='k')
    # now get theoretical bbox from markers interface
    origin_px = ax.transData.transform((0, 0))
    m_bbox = markers.MarkerStyle(marker).get_bbox(markersize, markeredgewidth)
    # convert from pt to pixel, and rename
    m_bottom_left, m_top_right = m_bbox.get_points() / 72.0 * fig.dpi
    top_right_px = origin_px + m_top_right
    bottom_left_px = origin_px + m_bottom_left
    right, top = ax.transData.inverted().transform(top_right_px)
    left, bottom = ax.transData.inverted().transform(bottom_left_px)
    # now draw that bbox in green
    ax.plot([left, right], [top, top], 'g')
    ax.plot([left, right], [bottom, bottom], 'g')
    ax.plot([right, right], [bottom, top], 'g')
    ax.plot([left, left], [bottom, top], 'g')

@image_comparison(baseline_images=['marker_bbox_star'],
                  extensions=['png'])
def test_marker_bbox_star():
    _draw_marker_outlined('*', markeredgewidth=20)

@image_comparison(baseline_images=['marker_bbox_pentagon'],
                  extensions=['png'])
def test_marker_bbox_pentagon():
    _draw_marker_outlined('p', markeredgewidth=20)

# we store some geometrical information about each marker to track how its
# size scales with increased "edge" thickness
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
_tri_tip_angle = 2*np.arctan(1/2)
_caret_side_angle = np.arctan(3/2)
_caret_tip_angle = 2*np.arctan(2/3)
# half the edge length of the smaller pentagon over the difference between the
# larger pentagon's circumcribing radius and the smaller pentagon's inscribed
# radius #TODO this formula has typo somewhere....
# _star_tip_angle = 2*np.arctan2((1/4)*np.sqrt((5 - np.sqrt(5))/2),
#                                1 - np.sqrt((3 + np.sqrt(5))/32))
_star_tip_angle = 0.6283185056636065
# reusable corner types
_flat_side = PathEndAngle(0, 0)
_normal_line = PathEndAngle(np.pi/2, None)
_normal_right_angle = PathEndAngle(np.pi/2, np.pi/2)
_tri_side = PathEndAngle(np.pi/2 - _tri_side_angle/2, _tri_side_angle)
_tri_tip = PathEndAngle(np.pi/2, _tri_tip_angle)
_caret_bottom = PathEndAngle(_caret_side_angle, None)
_caret_side = PathEndAngle(np.pi/2 - _caret_side_angle, None)
_caret_tip = PathEndAngle(np.pi/2, _caret_tip_angle)
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
    'v': BoxSides(_flat_side, _tri_tip, _tri_side, _tri_side),
    '^': BoxSides(_tri_tip, _flat_side, _tri_side, _tri_side),
    '<': BoxSides(_tri_side, _tri_side, _tri_tip, _flat_side),
    '>': BoxSides(_tri_side, _tri_side, _flat_side, _tri_tip),
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
    # carets missing the edge opposite their "tip", different size than tri's
    CARETLEFT: BoxSides(_caret_side, _caret_side, _caret_tip, _caret_bottom),
    CARETRIGHT: BoxSides(_caret_side, _caret_side, _caret_bottom, _caret_tip),
    CARETUP: BoxSides(_caret_tip, _caret_bottom, _caret_side, _caret_side),
    CARETDOWN: BoxSides(_caret_bottom, _caret_tip, _caret_side, _caret_side),
    CARETLEFTBASE: BoxSides(_caret_side, _caret_side, _caret_tip,
                            _caret_bottom),
    CARETRIGHTBASE: BoxSides(_caret_side, _caret_side, _caret_bottom,
                             _caret_tip),
    CARETUPBASE: BoxSides(_caret_tip, _caret_bottom, _caret_side, _caret_side),
    CARETDOWNBASE: BoxSides(_caret_bottom, _caret_tip, _caret_side,
                            _caret_side),
    '': BoxSides(None, None, None, None),
    ' ': BoxSides(None, None, None, None),
    'None': BoxSides(None, None, None, None),
    None: BoxSides(None, None, None, None),
}

def _get_bbox_path_end_angle(marker, markersize, markeredgewidth=0):
    """Get size of bbox if marker is centered at origin.

    For markers with no edge, this is just the same bbox as that of the
    transformed marker path, but how much extra extent is added by an edge
    is a function of the angle of the path at its own (the path's own)
    boundary.

    Parameters
    ----------
    markersize : float
        "Size" of the marker, in points.

    markeredgewidth : float, optional, default: 0
        Width, in points, of the stroke used to create the marker's edge.

    Returns
    -------
    bbox : matplotlib.transforms.Bbox
        The extents of the marker including its edge (in points) if it were
        centered at (0,0).

    """
    # if the marker is of size zero, the stroke's width doesn't matter,
    # there is no stroke so the bbox is trivial
    if np.isclose(markersize, 0):
        return Bbox([[0, 0], [0, 0]])
    unit_path = marker._transform.transform_path(marker._path)
    unit_bbox = unit_path.get_extents()
    scale = Affine2D().scale(markersize)
    [[left, bottom], [right, top]] = scale.transform(unit_bbox)
    angles = _edge_angles[marker._marker]
    left -= _get_padding_due_to_angle(markeredgewidth, angles.left,
                                        marker._joinstyle, marker._capstyle)
    bottom -= _get_padding_due_to_angle(markeredgewidth, angles.bottom,
                                        marker._joinstyle, marker._capstyle)
    right += _get_padding_due_to_angle(markeredgewidth, angles.right,
                                        marker._joinstyle, marker._capstyle)
    top += _get_padding_due_to_angle(markeredgewidth, angles.top,
                                        marker._joinstyle, marker._capstyle)
    return Bbox.from_extents(left, bottom, right, top)
