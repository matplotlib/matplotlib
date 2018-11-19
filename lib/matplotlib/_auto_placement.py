"""
Utility functions to find the best place in a plot for a rectangular box,
e.g. a legend or an annotation.
"""

import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox


def find_best_position(ax, width, height, consider):
    """
    Determine the best location to place the object.

    *consider* is a list of ``(x, y)`` pairs to consider as a potential
    lower-left corner of the legend. All are display coords.
    """
    verts, bboxes, lines, offsets = _plot_cover_data(ax)

    candidates = []

    for idx, (l, b) in enumerate(consider):
        box = Bbox.from_bounds(l, b, width, height)
        badness = 0
        # XXX TODO: If markers are present, it would be good to
        # take them into account when checking vertex overlaps in
        # the next line.
        badness = (box.count_contains(verts)
                   + box.count_contains(offsets)
                   + box.count_overlaps(bboxes)
                   + sum(line.intersects_bbox(box, filled=False)
                         for line in lines))
        if badness == 0:
            return l, b
        # Include the index to favor lower codes in case of a tie.
        candidates.append((badness, idx, (l, b)))

    _, _, (l, b) = min(candidates)
    return l, b


def _plot_cover_data(ax):
    """
    Returns list of vertices and extents covered by the plot.

    Returns a four long list.

    First element is a list of (x, y) vertices (in
    display-coordinates) covered by all the lines and line
    collections, in the legend's handles.

    Second element is a list of bounding boxes for all the patches in
    the plots's handles.

    Third element is a list of lines in the plot.

    Fourth element is a list of offsets for collections in the plot.
    """

    bboxes = []
    lines = []
    offsets = []

    for handle in ax.lines:
        assert isinstance(handle, Line2D)
        path = handle.get_path()
        trans = handle.get_transform()
        tpath = trans.transform_path(path)
        lines.append(tpath)

    for handle in ax.patches:
        assert isinstance(handle, Patch)

        if isinstance(handle, Rectangle):
            transform = handle.get_data_transform()
            bboxes.append(handle.get_bbox().transformed(transform))
        else:
            transform = handle.get_transform()
            bboxes.append(handle.get_path().get_extents(transform))

    for handle in ax.collections:
        transform, transOffset, hoffsets, paths = handle._prepare_points()

        if len(hoffsets):
            for offset in transOffset.transform(hoffsets):
                offsets.append(offset)

    try:
        vertices = np.concatenate([l.vertices for l in lines])
    except ValueError:
        vertices = np.array([])

    return [vertices, bboxes, lines, offsets]
