import numpy as np
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, knownfailureif

def test_connect_with_lines():
    M, L, C = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
    C3, C4 = Path.CURVE3, Path.CURVE4
    list_segments = lambda p: zip(*list(p.iter_segments()))

    points1 = np.asarray([[3,1], [4,1], [5,9], [2,6], [5,3]])
    points2 = np.asarray([[2,7], [1,8], [2,8], [9,9]])
    p1 = Path(points1, [M, L, C4, C4, C4])
    p2 = Path(points2, [M, C3, C3, C])

    p12 = Path.connect_with_lines((p1, p2))
    v12, c12 = list_segments(p12)
    # the vertex lists are of varying widths...
    check = lambda x, y: np.allclose(list(cbook.flatten(x)),
                                     list(cbook.flatten(y)))
    assert check(v12, [points1, points2])
    # we input C4, C4, C4 but get back just one C4
    assert check(c12, [M, L, C4,  L, C3, C])

    p1221 = Path.connect_with_lines((p12, p2, p1))
    v1221, c1221 = list_segments(p1221)
    assert check(v1221, [points1, points2, points2, points1])
    assert check(c1221, [M, L, C4,
                         L, C3, C,
                         L, C3, C,
                         L, L, C4])

@image_comparison(baseline_images=["reverse_path"])
def test_reverse_path():
    M, L, C4 = Path.MOVETO, Path.LINETO, Path.CURVE4
    p1 = Path([[1,-2], [2,0], [3,1]],
              [M,     L,     L])
    p2 = Path.arc(-45, 135)
    p3 = Path([[-2, 3], [-3,0], [-1, -1]],
              [M,       L,      L])

    fig = plt.figure()
    def plot(n, path):
        ax = fig.add_subplot(2, 2, n)
        ax.add_patch(PathPatch(path))
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticks([])
        ax.set_yticks([])

    plot(1, Path.connect_with_lines((p1, p2, p3)))
    plot(2, Path.connect_with_lines((p1, p2.reverse_path(), p3)))
    plot(3, Path.connect_with_lines((p1.reverse_path(), p2, p3.reverse_path())))
    plot(4, Path.connect_with_lines((p1, p2, p3)).reverse_path())
