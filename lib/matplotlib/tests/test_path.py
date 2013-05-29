from matplotlib.path import Path
from nose.tools import assert_raises


def test_readonly_path():
    path = Path.unit_circle()

    def modify_vertices():
        path.vertices = path.vertices * 2.0

    assert_raises(AttributeError, modify_vertices)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
