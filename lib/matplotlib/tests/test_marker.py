import numpy as np
from matplotlib import markers
from matplotlib.path import Path

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
