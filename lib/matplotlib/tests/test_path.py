from matplotlib.path import Path
from nose.tools import assert_raises

def test_readonly_path():
    path = Path.unit_circle()

    with assert_raises(AttributeError):
        path.vertices = path.vertices * 2.0
