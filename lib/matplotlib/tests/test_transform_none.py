from nose.tools import assert_equal

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.collections import Collection
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import IdentityTransform
from matplotlib.streamplot import streamplot
import numpy as np


def assert_transforms_equal(t1, t2):
    assert_equal(t1.to_values(), t2.to_values())


def test_transform_none():
    """
    Supplying transform=None to an artist should use device
    coordinates (i.e. transforms.IdentityTransform).
    """

    ax = plt.gca()

    #artists
    t = IdentityTransform()
    a = Artist()
    a.set_transform(None)
    assert_transforms_equal(a.get_transform(), t)

    a = Artist()
    a.update({'transform': None})
    assert_transforms_equal(a.get_transform(), t)

    e = Ellipse((0, 0), width=100, height=100, transform=None)
    assert_transforms_equal(Artist.get_transform(e), t)

    b = Rectangle((0, 0), width=40, height=40, transform=None, fc='r')
    assert_transforms_equal(Artist.get_transform(b), t)

    line, = ax.plot([1], [2], transform=None)
    assert_transforms_equal(line.get_transform(), t)

    #collections
    c = Collection()
    c.set_transform(None)
    assert_transforms_equal(c.get_transform(), t)

    points = ax.scatter([0], [1], transform=None)
    assert_transforms_equal(points.get_transform(), t)

    #misc
    x = np.array([1, 2, 3])
    y = x
    u = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    v = u
    s1 = streamplot(ax, x, y, u, v, transform=None)
    s2 = ax.streamplot(x, y, u, -v, color='m', transform=None)

    assert_transforms_equal(s1.lines.get_transform(), t)
    assert_transforms_equal(s2.lines.get_transform(), t)

    assert_transforms_equal(s1.arrows.get_transform(), t)
    assert_transforms_equal(s2.arrows.get_transform(), t)
