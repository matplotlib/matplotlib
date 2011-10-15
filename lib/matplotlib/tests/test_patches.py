import numpy as np
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.patches as patches
from matplotlib import pyplot as plt

def mk_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def plot_poly_and_rounded(ax, xy, radii, kwargs_poly={}, kwargs_rounded={}):
    poly = patches.Polygon(xy, **kwargs_poly)
    rounded = patches.RoundedPolygon(xy, radii, **kwargs_rounded)
    ax.add_patch(poly)
    ax.add_patch(rounded)

@image_comparison(baseline_images=["rounded_polygons", "rounded_nonconvex_1",
                                   "rounded_nonconvex_2", "rounded_nonconvex_3"])
def test_rounded_polygon():
    kwargs_poly = {'fc': 'None', 'ec': 'r', 'lw': 2, 'alpha': 0.5}
    kwargs_rounded = {'fc': 'None', 'ec': 'b', 'lw': 2, 'alpha': 0.5}
    ax = mk_axis()
    plot_poly_and_rounded(ax,
                          ((-2,-2),(2,-2),(2,2),(-2,2)),
                          (0, 0.5, 1, 1.5),
                          kwargs_poly,
                          kwargs_rounded)
    plot_poly_and_rounded(ax,
                          ((2,-3),(8,-3),(5,5)),
                          1,
                          kwargs_poly,
                          kwargs_rounded)
    ax.set_xlim(-3,9)
    ax.set_ylim(-5,7)

    # draw a star-like nonconvex polygon
    kwargs_rounded['fc'] = '#a0a0f0'
    ax = mk_axis()
    n = 10
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    inner_circle = 0.7 * np.c_[np.cos(t[0::2]), np.sin(t[0::2])]
    outer_circle = 1.3 * np.c_[np.cos(t[1::2]), np.sin(t[1::2])]
    xy = np.c_[inner_circle, outer_circle].reshape((n,2))
    plot_poly_and_rounded(ax, xy, 0.1, kwargs_poly, kwargs_rounded)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)

    # draw it in the other direction, should look identical
    ax = mk_axis()
    plot_poly_and_rounded(ax, xy[::-1], 0.1, kwargs_poly, kwargs_rounded)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)

    # shuffle the points, should look vaguely like a crescent
    ax = mk_axis()
    xy = np.r_[inner_circle, outer_circle[::-1,:]]
    plot_poly_and_rounded(ax, xy, 0.1, kwargs_poly, kwargs_rounded)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
