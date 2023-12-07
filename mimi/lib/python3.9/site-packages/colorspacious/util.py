# This file is part of colorspacious
# Copyright (C) 2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

def stacklast(*arrs):
    arrs = [np.asarray(arr)[..., np.newaxis] for arr in arrs]
    return np.concatenate(arrs, axis=-1)

def test_stacklast():
    assert np.array_equal(stacklast([1, 2, 3], [4, 5, 6]),
                          [[1, 4], [2, 5], [3, 6]])
    assert stacklast(np.zeros((3, 4)), np.zeros((3, 4))).shape == (3, 4, 2)

# Using color conventions: degrees 0-360
def color_cart2polar(a, b):
    h_rad = np.arctan2(b, a)
    h = np.rad2deg(h_rad) % 360
    r = np.hypot(a, b)
    return (r, h)

def test_color_cart2polar():
    assert np.allclose(color_cart2polar(0, 1), (1, 90))
    assert np.allclose(color_cart2polar(0, 10), (10, 90))
    assert np.allclose(color_cart2polar(-1, 0), (1, 180))

    assert np.allclose(color_cart2polar([0, 0, -1], [1, 10, 0]),
                       ([1, 10, 1], [90, 90, 180]))

def color_polar2cart(r, h):
    h_rad = np.deg2rad(h)
    return (r * np.cos(h_rad), r * np.sin(h_rad))

def test_color_polar2cart():
    assert np.allclose(color_polar2cart(1, 90), (0, 1))
    assert np.allclose(color_polar2cart(10, 90), (0, 10))
    assert np.allclose(color_polar2cart(1, 180), (-1, 0))

    assert np.allclose(color_polar2cart([1, 10, 1], [90, 90, 180]),
                       ([0, 0, -1], [1, 10, 0]))
