from copy import copy, deepcopy

from numpy.testing import assert_array_equal
from matplotlib.textpath import TextPath


def test_set_size():
    path = TextPath((0, 0), ".")
    _size = path.get_size()
    verts = path.vertices.copy()
    codes = path.codes.copy()
    path.set_size(20)
    assert_array_equal(verts/_size*path.get_size(), path.vertices)
    assert_array_equal(codes, path.codes)


def test_deepcopy():
    # Should not raise any error
    path = TextPath((0, 0), ".")
    path_copy = deepcopy(path)
    assert isinstance(path_copy, TextPath)
    assert path is not path_copy
    assert path.vertices is not path_copy.vertices
    assert path.codes is not path_copy.codes


def test_copy():
    # Should not raise any error
    path = TextPath((0, 0), ".")
    path_copy = copy(path)
    assert path is not path_copy
    assert path.vertices is path_copy.vertices
    assert path.codes is path_copy.codes
    path = TextPath((0, 0), ".")
    path_copy = path.copy()
    assert path is not path_copy
    assert path.vertices is path_copy.vertices
    assert path.codes is path_copy.codes
