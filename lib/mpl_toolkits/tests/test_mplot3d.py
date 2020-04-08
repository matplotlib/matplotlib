import functools

import pytest

from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np


mpl3d_image_comparison = functools.partial(
    image_comparison, remove_text=True, style='default')


def test_aspect_equal_error():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    with pytest.raises(NotImplementedError):
        ax.set_aspect('equal')


@mpl3d_image_comparison(['bar3d.png'])
def test_bar3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.arange(20)
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', align='edge', color=cs, alpha=0.8)


@mpl3d_image_comparison(['bar3d_shaded.png'])
def test_bar3d_shaded():
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d + 1  # Avoid triggering bug with zero-depth boxes.

    views = [(-60, 30), (30, 30), (30, -30), (120, -30)]
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(
        1, len(views),
        subplot_kw=dict(projection='3d')
    )
    for ax, (azim, elev) in zip(axs, views):
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        ax.view_init(azim=azim, elev=elev)
    fig.canvas.draw()


@mpl3d_image_comparison(['bar3d_notshaded.png'])
def test_bar3d_notshaded():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d
    ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()


def test_bar3d_lightsource():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ls = mcolors.LightSource(azdeg=0, altdeg=90)

    length, width = 3, 4
    area = length * width

    x, y = np.meshgrid(np.arange(length), np.arange(width))
    x = x.ravel()
    y = y.ravel()
    dz = x + y

    color = [cm.coolwarm(i/area) for i in range(area)]

    collection = ax.bar3d(x=x, y=y, z=0,
                          dx=1, dy=1, dz=dz,
                          color=color, shade=True, lightsource=ls)

    # Testing that the custom 90° lightsource produces different shading on
    # the top facecolors compared to the default, and that those colors are
    # precisely the colors from the colormap, due to the illumination parallel
    # to the z-axis.
    np.testing.assert_array_equal(color, collection._facecolors3d[1::6])


@mpl3d_image_comparison(['contour3d.png'])
def test_contour3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d.png'])
def test_contourf3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d_fill.png'])
def test_contourf3d_fill():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X.clip(0, 0)
    # This produces holes in the z=0 surface that causes rendering errors if
    # the Poly3DCollection is not aware of path code information (issue #4784)
    Z[::5, ::5] = 0.1
    ax.contourf(X, Y, Z, offset=0, levels=[-0.1, 0], cmap=cm.coolwarm)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)


@mpl3d_image_comparison(['tricontour.png'])
def test_tricontour():
    fig = plt.figure()

    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x**2 + y**2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)


@mpl3d_image_comparison(['lines3d.png'])
def test_lines3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)


@check_figures_equal(extensions=["png"])
def test_plot_scalar(fig_test, fig_ref):
    ax1 = fig_test.gca(projection='3d')
    ax1.plot([1], [1], "o")
    ax2 = fig_ref.gca(projection='3d')
    ax2.plot(1, 1, "o")


@mpl3d_image_comparison(['mixedsubplot.png'])
def test_mixedsubplots():
    def f(t):
        return np.cos(2*np.pi*t) * np.exp(-t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    R = np.hypot(X, Y)
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rcount=40, ccount=40,
                    linewidth=0, antialiased=False)

    ax.set_zlim3d(-1, 1)


@check_figures_equal(extensions=['png'])
def test_tight_layout_text(fig_test, fig_ref):
    # text is currently ignored in tight layout. So the order of text() and
    # tight_layout() calls should not influence the result.
    ax1 = fig_test.gca(projection='3d')
    ax1.text(.5, .5, .5, s='some string')
    fig_test.tight_layout()

    ax2 = fig_ref.gca(projection='3d')
    fig_ref.tight_layout()
    ax2.text(.5, .5, .5, s='some string')


@mpl3d_image_comparison(['scatter3d.png'])
def test_scatter3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               c='r', marker='o')
    x = y = z = np.arange(10, 20)
    ax.scatter(x, y, z, c='b', marker='^')
    z[-1] = 0  # Check that scatter() copies the data.


@mpl3d_image_comparison(['scatter3d_color.png'])
def test_scatter3d_color():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               color='r', marker='o')
    ax.scatter(np.arange(10, 20), np.arange(10, 20), np.arange(10, 20),
               color='b', marker='s')


@mpl3d_image_comparison(['plot_3d_from_2d.png'], tol=0.01)
def test_plot_3d_from_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(0, 5)
    ys = np.arange(5, 10)
    ax.plot(xs, ys, zs=0, zdir='x')
    ax.plot(xs, ys, zs=0, zdir='y')


@mpl3d_image_comparison(['surface3d.png'])
def test_surface3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.hypot(X, Y)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap=cm.coolwarm,
                           lw=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)


@mpl3d_image_comparison(['surface3d_shaded.png'])
def test_surface3d_shaded():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    color=[0.25, 1, 0.25], lw=1, antialiased=False)
    ax.set_zlim(-1.01, 1.01)


@mpl3d_image_comparison(['text3d.png'], remove_text=False)
def test_text3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    ax.text(1, 1, 1, "red", color='red')
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


@mpl3d_image_comparison(['trisurf3d.png'], tol=0.03)
def test_trisurf3d():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)


@mpl3d_image_comparison(['trisurf3d_shaded.png'], tol=0.03)
def test_trisurf3d_shaded():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, color=[1, 0.5, 0], linewidth=0.2)


@mpl3d_image_comparison(['wireframe3d.png'])
def test_wireframe3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=13)


@mpl3d_image_comparison(['wireframe3dzerocstride.png'])
def test_wireframe3dzerocstride():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=0)


@mpl3d_image_comparison(['wireframe3dzerorstride.png'])
def test_wireframe3dzerorstride():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=10)


def test_wireframe3dzerostrideraises():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=0, cstride=0)


def test_mixedsamplesraises():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=10, ccount=50)
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, Z, cstride=50, rcount=10)


@mpl3d_image_comparison(
    ['quiver3d.png', 'quiver3d_pivot_middle.png', 'quiver3d_pivot_tail.png'])
def test_quiver3d():
    x, y, z = np.ogrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    for pivot in ['tip', 'middle', 'tail']:
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, pivot=pivot, normalize=True)


@check_figures_equal(extensions=["png"])
def test_quiver3d_empty(fig_test, fig_ref):
    fig_ref.add_subplot(projection='3d')
    x = y = z = u = v = w = []
    ax = fig_test.add_subplot(projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


@mpl3d_image_comparison(['quiver3d_masked.png'])
def test_quiver3d_masked():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Using mgrid here instead of ogrid because masked_where doesn't
    # seem to like broadcasting very much...
    x, y, z = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]

    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)

    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


@mpl3d_image_comparison(['poly3dcollection_closed.png'])
def test_poly3dcollection_closed():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k',
                                facecolor=(1, 0.5, 0.5, 0.5), closed=False)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)


def test_poly_collection_2d_to_3d_empty():
    poly = PolyCollection([])
    art3d.poly_collection_2d_to_3d(poly)
    assert isinstance(poly, art3d.Poly3DCollection)
    assert poly.get_paths() == []


@mpl3d_image_comparison(['poly3dcollection_alpha.png'])
def test_poly3dcollection_alpha():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1), closed=True)
    c1.set_alpha(0.5)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k',
                                facecolor=(1, 0.5, 0.5), closed=False)
    c2.set_alpha(0.5)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)


@mpl3d_image_comparison(['axes3d_labelpad.png'], remove_text=False)
def test_axes3d_labelpad():
    fig = plt.figure()
    ax = Axes3D(fig)
    # labelpad respects rcParams
    assert ax.xaxis.labelpad == mpl.rcParams['axes.labelpad']
    # labelpad can be set in set_label
    ax.set_xlabel('X LABEL', labelpad=10)
    assert ax.xaxis.labelpad == 10
    ax.set_ylabel('Y LABEL')
    ax.set_zlabel('Z LABEL')
    # or manually
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = -40

    # Tick labels also respect tick.pad (also from rcParams)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_pad(tick.get_pad() - i * 5)


@mpl3d_image_comparison(['axes3d_cla.png'], remove_text=False)
def test_axes3d_cla():
    # fixed in pull request 4553
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_axis_off()
    ax.cla()  # make sure the axis displayed is 3D (not 2D)


@mpl3d_image_comparison(['axes3d_rotated.png'], remove_text=False)
def test_axes3d_rotated():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(90, 45)  # look down, rotated. Should be square


def test_plotsurface_1d_raises():
    x = np.linspace(0.5, 10, num=100)
    y = np.linspace(0.5, 10, num=100)
    X, Y = np.meshgrid(x, y)
    z = np.random.randn(100)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, z)


def _test_proj_make_M():
    # eye point
    E = np.array([1000, -1000, 2000])
    R = np.array([100, 100, 100])
    V = np.array([0, 0, 1])
    viewM = proj3d.view_transformation(E, R, V)
    perspM = proj3d.persp_transformation(100, -100)
    M = np.dot(perspM, viewM)
    return M


def test_proj_transform():
    M = _test_proj_make_M()

    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    ixs, iys, izs = proj3d.inv_transform(txs, tys, tzs, M)

    np.testing.assert_almost_equal(ixs, xs)
    np.testing.assert_almost_equal(iys, ys)
    np.testing.assert_almost_equal(izs, zs)


def _test_proj_draw_axes(M, s=1, *args, **kwargs):
    xs = [0, s, 0, 0]
    ys = [0, 0, s, 0]
    zs = [0, 0, 0, s]
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    o, ax, ay, az = zip(txs, tys)
    lines = [(o, ax), (o, ay), (o, az)]

    fig, ax = plt.subplots(*args, **kwargs)
    linec = LineCollection(lines)
    ax.add_collection(linec)
    for x, y, t in zip(txs, tys, ['o', 'x', 'y', 'z']):
        ax.text(x, y, t)

    return fig, ax


@mpl3d_image_comparison(['proj3d_axes_cube.png'])
def test_proj_axes_cube():
    M = _test_proj_make_M()

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=400)

    ax.scatter(txs, tys, c=tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)


@mpl3d_image_comparison(['proj3d_axes_cube_ortho.png'])
def test_proj_axes_cube_ortho():
    E = np.array([200, 100, 100])
    R = np.array([0, 0, 0])
    V = np.array([0, 0, 1])
    viewM = proj3d.view_transformation(E, R, V)
    orthoM = proj3d.ortho_transformation(-1, 1)
    M = np.dot(orthoM, viewM)

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 100
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 100
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 100

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=150)

    ax.scatter(txs, tys, s=300-tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)


def test_rot():
    V = [1, 0, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [1, 0, 0, 1])

    V = [0, 1, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [0, np.sqrt(3) / 2, 0.5, 1])


def test_world():
    xmin, xmax = 100, 120
    ymin, ymax = -100, 100
    zmin, zmax = 0.1, 0.2
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    np.testing.assert_allclose(M,
                               [[5e-2, 0, 0, -5],
                                [0, 5e-3, 0, 5e-1],
                                [0, 0, 1e1, -1],
                                [0, 0, 0, 1]])


@mpl3d_image_comparison(['proj3d_lines_dists.png'])
def test_lines_dists():
    fig, ax = plt.subplots(figsize=(4, 6), subplot_kw=dict(aspect='equal'))

    xs = (0, 30)
    ys = (20, 150)
    ax.plot(xs, ys)
    p0, p1 = zip(xs, ys)

    xs = (0, 0, 20, 30)
    ys = (100, 150, 30, 200)
    ax.scatter(xs, ys)

    dist = proj3d._line2d_seg_dist(p0, p1, (xs[0], ys[0]))
    dist = proj3d._line2d_seg_dist(p0, p1, np.array((xs, ys)))
    for x, y, d in zip(xs, ys, dist):
        c = Circle((x, y), d, fill=0)
        ax.add_patch(c)

    ax.set_xlim(-50, 150)
    ax.set_ylim(0, 300)


def test_autoscale():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.margins(x=0, y=.1, z=.2)
    ax.plot([0, 1], [0, 1], [0, 1])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.2, 1.2)
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.4, 2.4)


@mpl3d_image_comparison(['axes3d_ortho.png'], remove_text=False)
def test_axes3d_ortho():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_proj_type('ortho')


@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [
    ('set_xlim3d', 'left'),
    ('set_xlim3d', 'right'),
    ('set_ylim3d', 'bottom'),
    ('set_ylim3d', 'top'),
    ('set_zlim3d', 'bottom'),
    ('set_zlim3d', 'top'),
])
def test_invalid_axes_limits(setter, side, value):
    limit = {side: value}
    fig = plt.figure()
    obj = fig.add_subplot(111, projection='3d')
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)


class TestVoxels:
    @mpl3d_image_comparison(['voxels-simple.png'])
    def test_simple(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 4, 3))
        voxels = (x == y) | (y == z)
        ax.voxels(voxels)

    @mpl3d_image_comparison(['voxels-edge-style.png'])
    def test_edge_style(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 5, 4))
        voxels = ((x - 2)**2 + (y - 2)**2 + (z-1.5)**2) < 2.2**2
        v = ax.voxels(voxels, linewidths=3, edgecolor='C1')

        # change the edge color of one voxel
        v[max(v.keys())].set_edgecolor('C2')

    @mpl3d_image_comparison(['voxels-named-colors.png'])
    def test_named_colors(self):
        """Test with colors set to a 3d object array of strings."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        voxels = voxels & ~(x * y * z < 1)
        colors = np.full((10, 10, 10), 'C0', dtype=np.object_)
        colors[(x < 5) & (y < 5)] = '0.25'
        colors[(x + z) < 10] = 'cyan'
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-rgb-data.png'])
    def test_rgb_data(self):
        """Test with colors set to a 4d float array of rgb data."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        colors = np.zeros((10, 10, 10, 3))
        colors[..., 0] = x / 9
        colors[..., 1] = y / 9
        colors[..., 2] = z / 9
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-alpha.png'])
    def test_alpha(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        v1 = x == y
        v2 = np.abs(x - y) < 2
        voxels = v1 | v2
        colors = np.zeros((10, 10, 10, 4))
        colors[v2] = [1, 0, 0, 0.5]
        colors[v1] = [0, 1, 0, 0.5]
        v = ax.voxels(voxels, facecolors=colors)

        assert type(v) is dict
        for coord, poly in v.items():
            assert voxels[coord], "faces returned for absent voxel"
            assert isinstance(poly, art3d.Poly3DCollection)

    @mpl3d_image_comparison(['voxels-xyz.png'], tol=0.01, remove_text=False)
    def test_xyz(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def midpoints(x):
            sl = ()
            for i in range(x.ndim):
                x = (x[sl + np.index_exp[:-1]] +
                     x[sl + np.index_exp[1:]]) / 2.0
                sl += np.index_exp[:]
            return x

        # prepare some coordinates, and attach rgb values to each
        r, g, b = np.indices((17, 17, 17)) / 16.0
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)

        # define a sphere about [0.5, 0.5, 0.5]
        sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

        # combine the color components
        colors = np.zeros(sphere.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc

        # and plot everything
        ax.voxels(r, g, b, sphere,
                  facecolors=colors,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                  linewidth=0.5)

    def test_calling_conventions(self):
        x, y, z = np.indices((3, 4, 5))
        filled = np.ones((2, 3, 4))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # all the valid calling conventions
        for kw in (dict(), dict(edgecolor='k')):
            ax.voxels(filled, **kw)
            ax.voxels(filled=filled, **kw)
            ax.voxels(x, y, z, filled, **kw)
            ax.voxels(x, y, z, filled=filled, **kw)

        # duplicate argument
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y, z, filled, filled=filled)
        # missing arguments
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y)
        # x, y, z are positional only - this passes them on as attributes of
        # Poly3DCollection
        with pytest.raises(AttributeError):
            ax.voxels(filled=filled, x=x, y=y, z=z)


def test_line3d_set_get_data_3d():
    x, y, z = [0, 1], [2, 3], [4, 5]
    x2, y2, z2 = [6, 7], [8, 9], [10, 11]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lines = ax.plot(x, y, z)
    line = lines[0]
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_data_3d(x2, y2, z2)
    np.testing.assert_array_equal((x2, y2, z2), line.get_data_3d())


@check_figures_equal(extensions=["png"])
def test_inverted(fig_test, fig_ref):
    # Plot then invert.
    ax = fig_test.add_subplot(projection="3d")
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])
    ax.invert_yaxis()
    # Invert then plot.
    ax = fig_ref.add_subplot(projection="3d")
    ax.invert_yaxis()
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])


def test_inverted_cla():
    # GitHub PR #5450. Setting autoscale should reset
    # axes to be non-inverted.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 1. test that a new axis is not inverted per default
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()
    ax.set_xlim(1, 0)
    ax.set_ylim(1, 0)
    ax.set_zlim(1, 0)
    assert ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    assert ax.zaxis_inverted()
    ax.cla()
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()


def test_ax3d_tickcolour():
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='z', colors='red')
    fig.canvas.draw()

    for tick in ax.xaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.yaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.zaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'


@check_figures_equal(extensions=["png"])
def test_ticklabel_format(fig_test, fig_ref):
    axs = fig_test.subplots(4, 5, subplot_kw={"projection": "3d"})
    for ax in axs.flat:
        ax.set_xlim(1e7, 1e7 + 10)
    for row, name in zip(axs, ["x", "y", "z", "both"]):
        row[0].ticklabel_format(
            axis=name, style="plain")
        row[1].ticklabel_format(
            axis=name, scilimits=(-2, 2))
        row[2].ticklabel_format(
            axis=name, useOffset=not mpl.rcParams["axes.formatter.useoffset"])
        row[3].ticklabel_format(
            axis=name, useLocale=not mpl.rcParams["axes.formatter.use_locale"])
        row[4].ticklabel_format(
            axis=name,
            useMathText=not mpl.rcParams["axes.formatter.use_mathtext"])

    def get_formatters(ax, names):
        return [getattr(ax, name).get_major_formatter() for name in names]

    axs = fig_ref.subplots(4, 5, subplot_kw={"projection": "3d"})
    for ax in axs.flat:
        ax.set_xlim(1e7, 1e7 + 10)
    for row, names in zip(
            axs, [["xaxis"], ["yaxis"], ["zaxis"], ["xaxis", "yaxis", "zaxis"]]
    ):
        for fmt in get_formatters(row[0], names):
            fmt.set_scientific(False)
        for fmt in get_formatters(row[1], names):
            fmt.set_powerlimits((-2, 2))
        for fmt in get_formatters(row[2], names):
            fmt.set_useOffset(not mpl.rcParams["axes.formatter.useoffset"])
        for fmt in get_formatters(row[3], names):
            fmt.set_useLocale(not mpl.rcParams["axes.formatter.use_locale"])
        for fmt in get_formatters(row[4], names):
            fmt.set_useMathText(
                not mpl.rcParams["axes.formatter.use_mathtext"])


@check_figures_equal(extensions=["png"])
def test_quiver3D_smoke(fig_test, fig_ref):
    pivot = "middle"
    # Make the grid
    x, y, z = np.meshgrid(
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.8)
    )
    u = v = w = np.ones_like(x)

    for fig, length in zip((fig_ref, fig_test), (1, 1.0)):
        ax = fig.gca(projection="3d")
        ax.quiver(x, y, z, u, v, w, length=length, pivot=pivot)


@image_comparison(["minor_ticks.png"], style="mpl20")
def test_minor_ticks():
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xticks([0.25], minor=True)
    ax.set_xticklabels(["quarter"], minor=True)
    ax.set_yticks([0.33], minor=True)
    ax.set_yticklabels(["third"], minor=True)
    ax.set_zticks([0.50], minor=True)
    ax.set_zticklabels(["half"], minor=True)
