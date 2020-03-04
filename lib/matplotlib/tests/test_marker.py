import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison

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
    fig, ax = plt.subplots(figsize=(fig_d,fig_d))
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
    origin_px = ax.transData.transform((0,0))
    m_bbox = markers.MarkerStyle(marker).get_centered_bbox(markersize, markeredgewidth)
    m_bottom_left, m_top_right = m_bbox.get_points()
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
