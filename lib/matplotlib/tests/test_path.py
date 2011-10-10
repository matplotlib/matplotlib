import numpy as np
import matplotlib.cbook as cbook
import matplotlib.path as path

def test_connect_with_lines():
    M, L, C = path.Path.MOVETO, path.Path.LINETO, path.Path.CLOSEPOLY
    C3, C4 = path.Path.CURVE3, path.Path.CURVE4
    list_segments = lambda p: zip(*list(p.iter_segments()))

    points1 = np.asarray([[3,1], [4,1], [5,9], [2,6], [5,3]])
    points2 = np.asarray([[2,7], [1,8], [2,8], [9,9]])
    p1 = path.Path(points1, [M, L, C4, C4, C4])
    p2 = path.Path(points2, [M, C3, C3, C])

    p12 = path.Path.connect_with_lines((p1, p2))
    v12, c12 = list_segments(p12)
    # the vertex lists are of varying widths...
    check = lambda x, y: np.allclose(list(cbook.flatten(x)),
                                     list(cbook.flatten(y)))
    assert check(v12, [points1, points2])
    # we input C4, C4, C4 but get back just one C4
    assert check(c12, [M, L, C4,  L, C3, C])

    p1221 = path.Path.connect_with_lines((p12, p2, p1))
    v1221, c1221 = list_segments(p1221)
    assert check(v1221, [points1, points2, points2, points1])
    assert check(c1221, [M, L, C4,
                         L, C3, C,
                         L, C3, C,
                         L, L, C4])
