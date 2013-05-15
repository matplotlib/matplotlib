from matplotlib.path import Path
from nose.tools import assert_raises

def test_readonly_path():
    def readonly():
        path = Path.unit_circle()
        path.vertices = path.vertices * 2.0

    assert_raises(AttributeError, readonly)
