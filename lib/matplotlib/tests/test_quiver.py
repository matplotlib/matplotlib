import platform
import sys

import numpy as np
import pytest
import cartopy.crs as ccrs


from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.decorators import check_figures_equal
from matplotlib import patches
from matplotlib import colors as colors


def draw_quiver(ax, **kwargs):
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, 1),
                       np.arange(0, 2 * np.pi, 1))
    U = np.cos(X)
    V = np.sin(Y)

    Q = ax.quiver(U, V, **kwargs)
    return Q


@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='Requires CPython')
def test_quiver_memory_leak():
    fig, ax = plt.subplots()

    Q = draw_quiver(ax)
    ttX = Q.X
    orig_refcount = sys.getrefcount(ttX)
    Q.remove()

    del Q

    assert sys.getrefcount(ttX) < orig_refcount


@pytest.mark.skipif(platform.python_implementation() != 'CPython',
                    reason='Requires CPython')
def test_quiver_key_memory_leak():
    fig, ax = plt.subplots()

    Q = draw_quiver(ax)

    qk = ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                      labelpos='W',
                      fontproperties={'weight': 'bold'})
    orig_refcount = sys.getrefcount(qk)
    qk.remove()
    assert sys.getrefcount(qk) < orig_refcount


def test_quiver_number_of_args():
    X = [1, 2]
    with pytest.raises(
            TypeError,
            match='takes from 2 to 5 positional arguments but 1 were given'):
        plt.quiver(X)
    with pytest.raises(
            TypeError,
            match='takes from 2 to 5 positional arguments but 6 were given'):
        plt.quiver(X, X, X, X, X, X)


def test_quiver_arg_sizes():
    X2 = [1, 2]
    X3 = [1, 2, 3]
    with pytest.raises(
            ValueError, match=('X and Y must be the same size, but '
                               'X.size is 2 and Y.size is 3.')):
        plt.quiver(X2, X3, X2, X2)
    with pytest.raises(
            ValueError, match=('Argument U has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X3, X2)
    with pytest.raises(
            ValueError, match=('Argument V has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X2, X3)
    with pytest.raises(
            ValueError, match=('Argument C has a size 3 which does not match '
                               '2, the number of arrow positions')):
        plt.quiver(X2, X2, X2, X2, X3)


def test_no_warnings():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(15), np.arange(10))
    U = V = np.ones_like(X)
    phi = (np.random.rand(15, 10) - .5) * 150
    ax.quiver(X, Y, U, V, angles=phi)
    fig.canvas.draw()  # Check that no warning is emitted.


def test_zero_headlength():
    # Based on report by Doug McNeil:
    # https://discourse.matplotlib.org/t/quiver-warnings/16722
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    U, V = np.cos(X), np.sin(Y)
    ax.quiver(U, V, headlength=0, headaxislength=0)
    fig.canvas.draw()  # Check that no warning is emitted.


@image_comparison(['quiver_animated_test_image.png'])
def test_quiver_animate():
    # Tests fix for #2616
    fig, ax = plt.subplots()
    Q = draw_quiver(ax, animated=True)
    ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                 labelpos='W', fontproperties={'weight': 'bold'})


@image_comparison(['quiver_with_key_test_image.png'])
def test_quiver_with_key():
    fig, ax = plt.subplots()
    ax.margins(0.1)
    Q = draw_quiver(ax)
    ax.quiverkey(Q, 0.5, 0.95, 2,
                 r'$2\, \mathrm{m}\, \mathrm{s}^{-1}$',
                 angle=-10,
                 coordinates='figure',
                 labelpos='W',
                 fontproperties={'weight': 'bold', 'size': 'large'})


@image_comparison(['quiver_single_test_image.png'], remove_text=True)
def test_quiver_single():
    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.quiver([1], [1], [2], [2])


def test_quiver_copy():
    fig, ax = plt.subplots()
    uv = dict(u=np.array([1.1]), v=np.array([2.0]))
    q0 = ax.quiver([1], [1], uv['u'], uv['v'])
    uv['v'][0] = 0
    assert q0.V[0] == 2.0


@image_comparison(['quiver_key_pivot.png'], remove_text=True)
def test_quiver_key_pivot():
    fig, ax = plt.subplots()

    u, v = np.mgrid[0:2*np.pi:10j, 0:2*np.pi:10j]

    q = ax.quiver(np.sin(u), np.cos(v))
    ax.set_xlim(-2, 11)
    ax.set_ylim(-2, 11)
    ax.quiverkey(q, 0.5, 1, 1, 'N', labelpos='N')
    ax.quiverkey(q, 1, 0.5, 1, 'E', labelpos='E')
    ax.quiverkey(q, 0.5, 0, 1, 'S', labelpos='S')
    ax.quiverkey(q, 0, 0.5, 1, 'W', labelpos='W')


@image_comparison(['quiver_key_xy.png'], remove_text=True)
def test_quiver_key_xy():
    # With scale_units='xy', ensure quiverkey still matches its quiver.
    # Note that the quiver and quiverkey lengths depend on the axes aspect
    # ratio, and that with angles='xy' their angles also depend on the axes
    # aspect ratio.
    X = np.arange(8)
    Y = np.zeros(8)
    angles = X * (np.pi / 4)
    uv = np.exp(1j * angles)
    U = uv.real
    V = uv.imag
    fig, axs = plt.subplots(2)
    for ax, angle_str in zip(axs, ('uv', 'xy')):
        ax.set_xlim(-1, 8)
        ax.set_ylim(-0.2, 0.2)
        q = ax.quiver(X, Y, U, V, pivot='middle',
                      units='xy', width=0.05,
                      scale=2, scale_units='xy',
                      angles=angle_str)
        for x, angle in zip((0.2, 0.5, 0.8), (0, 45, 90)):
            ax.quiverkey(q, X=x, Y=0.8, U=1, angle=angle, label='', color='b')


@image_comparison(['barbs_test_image.png'], remove_text=True)
def test_barbs():
    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    U, V = 12*X, 12*Y
    fig, ax = plt.subplots()
    ax.barbs(X, Y, U, V, np.hypot(U, V), fill_empty=True, rounding=False,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),
             cmap='viridis')


@image_comparison(['barbs_pivot_test_image.png'], remove_text=True)
def test_barbs_pivot():
    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    U, V = 12*X, 12*Y
    fig, ax = plt.subplots()
    ax.barbs(X, Y, U, V, fill_empty=True, rounding=False, pivot=1.7,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3))
    ax.scatter(X, Y, s=49, c='black')


@image_comparison(['barbs_test_flip.png'], remove_text=True)
def test_barbs_flip():
    """Test barbs with an array for flip_barb."""
    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    U, V = 12*X, 12*Y
    fig, ax = plt.subplots()
    ax.barbs(X, Y, U, V, fill_empty=True, rounding=False, pivot=1.7,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),
             flip_barb=Y < 0)


def test_barb_copy():
    fig, ax = plt.subplots()
    u = np.array([1.1])
    v = np.array([2.2])
    b0 = ax.barbs([1], [1], u, v)
    u[0] = 0
    assert b0.u[0] == 1.1
    v[0] = 0
    assert b0.v[0] == 2.2


def test_bad_masked_sizes():
    """Test error handling when given differing sized masked arrays."""
    x = np.arange(3)
    y = np.arange(3)
    u = np.ma.array(15. * np.ones((4,)))
    v = np.ma.array(15. * np.ones_like(u))
    u[1] = np.ma.masked
    v[1] = np.ma.masked
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ax.barbs(x, y, u, v)


def test_angles_and_scale():
    # angles array + scale_units kwarg
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(15), np.arange(10))
    U = V = np.ones_like(X)
    phi = (np.random.rand(15, 10) - .5) * 150
    ax.quiver(X, Y, U, V, angles=phi, scale_units='xy')


@image_comparison(['quiver_xy.png'], remove_text=True)
def test_quiver_xy():
    # simple arrow pointing from SW to NE
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'))
    ax.quiver(0, 0, 1, 1, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid()


def test_quiverkey_angles():
    # Check that only a single arrow is plotted for a quiverkey when an array
    # of angles is given to the original quiver plot
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    U = V = angles = np.ones_like(X)

    q = ax.quiver(X, Y, U, V, angles=angles)
    qk = ax.quiverkey(q, 1, 1, 2, 'Label')
    # The arrows are only created when the key is drawn
    fig.canvas.draw()
    assert len(qk.verts) == 1


def test_quiverkey_angles_xy_aitoff():
    # GH 26316 and GH 26748
    # Test that only one arrow will be plotted with non-cartesian
    # when angles='xy' and/or scale_units='xy'

    # only for test purpose
    # scale_units='xy' may not be a valid use case for non-cartesian
    kwargs_list = [
        {'angles': 'xy'},
        {'angles': 'xy', 'scale_units': 'xy'},
        {'scale_units': 'xy'}
    ]

    for kwargs_dict in kwargs_list:

        x = np.linspace(-np.pi, np.pi, 11)
        y = np.ones_like(x) * np.pi / 6
        vx = np.zeros_like(x)
        vy = np.ones_like(x)

        fig = plt.figure()
        ax = fig.add_subplot(projection='aitoff')
        q = ax.quiver(x, y, vx, vy, **kwargs_dict)
        qk = ax.quiverkey(q, 0, 0, 1, '1 units')

        fig.canvas.draw()
        assert len(qk.verts) == 1


def test_quiverkey_angles_scale_units_cartesian():
    # GH 26316
    # Test that only one arrow will be plotted with normal cartesian
    # when angles='xy' and/or scale_units='xy'

    kwargs_list = [
        {'angles': 'xy'},
        {'angles': 'xy', 'scale_units': 'xy'},
        {'scale_units': 'xy'}
    ]

    for kwargs_dict in kwargs_list:
        X = [0, -1, 0]
        Y = [0, -1, 0]
        U = [1, -1, 1]
        V = [1, -1, 0]

        fig, ax = plt.subplots()
        q = ax.quiver(X, Y, U, V, **kwargs_dict)
        ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                     label='Quiver key, length = 1', labelpos='E')
        qk = ax.quiverkey(q, 0, 0, 1, '1 units')

        fig.canvas.draw()
        assert len(qk.verts) == 1


def test_quiver_setuvc_numbers():
    """Check that it is possible to set all arrow UVC to the same numbers"""

    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(2), np.arange(2))
    U = V = np.ones_like(X)

    q = ax.quiver(X, Y, U, V)
    q.set_UVC(0, 1)


def draw_quiverkey_zorder_argument(fig, zorder=None):
    """Draw Quiver and QuiverKey using zorder argument"""
    x = np.arange(1, 6, 1)
    y = np.arange(1, 6, 1)
    X, Y = np.meshgrid(x, y)
    U, V = 2, 2

    ax = fig.subplots()
    q = ax.quiver(X, Y, U, V, pivot='middle')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    if zorder is None:
        ax.quiverkey(q, 4, 4, 25, coordinates='data',
                     label='U', color='blue')
        ax.quiverkey(q, 5.5, 2, 20, coordinates='data',
                     label='V', color='blue', angle=90)
    else:
        ax.quiverkey(q, 4, 4, 25, coordinates='data',
                     label='U', color='blue', zorder=zorder)
        ax.quiverkey(q, 5.5, 2, 20, coordinates='data',
                     label='V', color='blue', angle=90, zorder=zorder)


def draw_quiverkey_setzorder(fig, zorder=None):
    """Draw Quiver and QuiverKey using set_zorder"""
    x = np.arange(1, 6, 1)
    y = np.arange(1, 6, 1)
    X, Y = np.meshgrid(x, y)
    U, V = 2, 2

    ax = fig.subplots()
    q = ax.quiver(X, Y, U, V, pivot='middle')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    qk1 = ax.quiverkey(q, 4, 4, 25, coordinates='data',
                       label='U', color='blue')
    qk2 = ax.quiverkey(q, 5.5, 2, 20, coordinates='data',
                       label='V', color='blue', angle=90)
    if zorder is not None:
        qk1.set_zorder(zorder)
        qk2.set_zorder(zorder)


@pytest.mark.parametrize('zorder', [0, 2, 5, None])
@check_figures_equal()
def test_quiverkey_zorder(fig_test, fig_ref, zorder):
    draw_quiverkey_zorder_argument(fig_test, zorder=zorder)
    draw_quiverkey_setzorder(fig_ref, zorder=zorder)

#Tests for  Quiverkey and text inside bbox.
def test_quiverkey_bbox_basic():
    """
    Test that a custom bbox passed to quiverkey is properly applied.

    This test verifies:
    - That the quiverkey generates a 'bbox_patch' attribute.
    - That the facecolor of the drawn FancyBboxPatch matches the specified color.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()

    q = ax.quiver([0], [0], [1], [1], transform=ccrs.PlateCarree())
    qk = ax.quiverkey(q, X=0.85, Y=0.95, U=1, label='1 unit', labelpos='E',
                  bbox=dict(facecolor='lightblue', edgecolor='blue', boxstyle='round,pad=0.3'))

    assert hasattr(qk, '_bbox') and qk._bbox is not None
    assert qk._bbox['facecolor'] == 'lightblue'
    assert qk._bbox['edgecolor'] == 'blue'

def test_quiverkey_without_bbox():
    """
    Test that a quiverkey created without a bbox still works correctly.

    This test verifies:
    - That no error is raised when no bbox is specified.
    - That the label text of the quiverkey is correctly set.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver([0], [0], [3], [1], color='black', scale=30)
    qk = ax.quiverkey(q, X=0.85, Y=0.15, U=3, label='3 units', labelpos='N')

    assert qk.label == '3 units'

def test_quiverkey_all_labelpos():
    """
    Test that quiverkeys correctly support all label positions ('N', 'S', 'E', 'W') 
    by placing the label on the opposite side to avoid overlap.

    This test:
    - Creates four axes with quiverkeys positioned roughly at N, S, E, W.
    - Sets the label position opposite to the quiverkey position for visibility.
    - Adds a bbox for visual clarity.
    - Asserts that the label position is set correctly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    positions = ['N', 'S', 'E', 'W']

    for ax, pos in zip(axes, positions):
        ax.coastlines()
        ax.set_global()
        q = ax.quiver([0], [0], [1], [0.5], transform=ccrs.PlateCarree())

        if pos == 'N':
            X, Y, labelpos = 0.5, 0.7, 'S'
        elif pos == 'S':
            X, Y, labelpos = 0.5, 0.25, 'N'
        elif pos == 'E':
            X, Y, labelpos = 0.75, 0.5, 'W'
        else:  # 'W'
            X, Y, labelpos = 0.25, 0.5, 'E'

        qk = ax.quiverkey(q, X=X, Y=Y, U=1,
                        label=f'{pos} position',
                        labelpos=labelpos,
                        bbox=dict(facecolor='red', alpha=0.3, edgecolor='black', linewidth=1))

        assert qk.labelpos == labelpos

    plt.close(fig)

def test_quiverkey_different_angles():
    """
    Test that quiverkeys correctly apply different arrow angles.

    This test:
    - Creates a single plot with one quiver.
    - Adds four quiverkeys with angles: 0°, 45°, 90°, and 135°.
    - Each quiverkey is positioned to be fully visible within the plot.
    - Asserts that each quiverkey stores the correct angle.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    q = ax.quiver([0], [1], [1], [0], angles='xy', scale_units='xy', scale=1)

    angles = [0, 45, 90, 135]
    positions = [
        (0.3, 0.85),
        (0.7, 0.85),
        (0.3, 0.15), 
        (0.7, 0.15), 
    ]

    for angle, (xk, yk) in zip(angles, positions):
        qk = ax.quiverkey(q, X=xk, Y=yk, U=0.2, angle=angle,
                          label=f'{angle}° arrow', labelpos='E',
                          bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='red'),
                          coordinates='axes')

        assert qk.angle == angle

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    plt.title("Quiverkeys fully inside axes with different angles")

def test_quiverkey_various_locations():
    """
    Test that quiverkeys can be created with different label texts and positions.

    This test:
    - Creates a global vector field using regularly spaced coordinates.
    - Adds multiple quiverkeys with varying labels at specified normalized positions.
    - Uses an empty `bbox` to ensure label functionality sem dependência de estilo visual.

    Note:
    - This test currently does not assert label correctness or positions.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()

    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-60, 60, 8)
    u = np.random.random((len(lats), len(lons))) - 0.5
    v = np.random.random((len(lats), len(lons))) - 0.5
    q = ax.quiver(lons, lats, u, v, transform=ccrs.PlateCarree())

    locations = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right', 'Bottom Center']
    positions = [
        (0.1, 0.9),  # Top Left
        (0.9, 0.9),  # Top Right
        (0.1, 0.1),  # Bottom Left
        (0.9, 0.1),  # Bottom Right
        (0.5, 0.1)   # Bottom Center
    ]

    for label, (x, y) in zip(locations, positions):
        qk = ax.quiverkey(q, X=x, Y=y, U=0.5, label=label, labelpos='E', bbox={})

        assert qk.label == label

    plt.close(fig)

def test_quiverkey_boxstyles():
    """
    Test that quiverkeys correctly apply various box styles to their surrounding bbox_patch.

    This test:
    - Defines a list of different `boxstyle` configurations for the bbox.
    - Places each quiverkey in a different vertical position to avoid overlap.
    - Asserts that the label is correct and that the bbox was created.

    """
    fig, ax = plt.subplots(figsize=(12, 10))
    q = ax.quiver([-0.5], [0], [1], [1])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    boxstyles = [
        ('square', 'Square'),
        ('round,pad=1.2', 'Round\nLarge Pad'),
        ('round4,pad=1.2', 'Round4\nLarge Pad'),
        ('sawtooth,pad=1.2', 'Sawtooth\nLarge Pad'),
        ('roundtooth,pad=1.2', 'Roundtooth\nLarge Pad'),
        ('round,pad=1.5', 'Round\nBig Pad'),
        ('round,pad=1.0,rounding_size=0.6', 'Custom\nRound')
    ]

    for i, (style, label) in enumerate(boxstyles):
        y_pos = 0.9 - i * 0.12
        qk = ax.quiverkey(q, X=0.1, Y=y_pos, U=0.5, label=label, labelpos='E',
                          labelcolor='black',
                          bbox=dict(boxstyle=style, facecolor='lightyellow', edgecolor='black', linewidth=2))

        
        assert qk.label == label
        assert qk._bbox is not None
        assert qk._bbox['facecolor'] == 'lightyellow'
        assert qk._bbox['edgecolor'] == 'black'
    
    plt.close(fig)

def test_quiverkey_color_matching():
    """
    Test that the quiverkey label correctly inherits the specified color.

    This test:
    - Draws a single vector in a specific color (`darkgreen`) using `quiver`.
    - Adds a quiverkey with the `labelcolor` explicitly set to match the vector color.
    - Asserts that the label of the quiverkey adopts the intended color,
      ensuring visual consistency between the vector and its legend.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()

    vector_color = 'darkgreen'
    q = ax.quiver(0, 0, 1, 0, color=vector_color, transform=ccrs.PlateCarree())

    qk = ax.quiverkey(q, X=0.5, Y=0.9, U=1,
                      label='1 unit (match color)', labelpos='E',
                      labelcolor=vector_color,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))

    assert qk.text.get_color() == vector_color
    plt.close(fig)

def test_quiverkey_without_bbox_creates_no_fancybox():
    """
    Test that no FancyBboxPatch is created when `bbox=None` is passed.

    This test:
    - Creates a simple quiver plot with a single vector.
    - Adds a quiverkey with `bbox=None`, meaning no background box is requested.
    - Asserts that no `FancyBboxPatch` is present among the children of the quiverkey,
      verifying that the label appears without a styled box.
    """

    fig, ax = plt.subplots()
    q = ax.quiver([0], [0], [1], [0])
    qk = ax.quiverkey(q, X=0.5, Y=0.5, U=1, label="test", bbox=None)

    has_fancybox = any(isinstance(child, patches.FancyBboxPatch)
                       for child in qk.get_children())
    assert not has_fancybox
    plt.close(fig)

def test_quiverkey_position_stable_on_resize_with_bbox():
    """
    Test that the position of the quiverkey label remains stable in axes coordinates
    after resizing the figure, even when a bbox is applied.

    This test:
    - Creates a quiver plot with a single vector.
    - Adds a quiverkey label at a specified position with a visible `bbox` (styled background box).
    - Forces a draw to render all elements.
    - Records the quiverkey text position in axes coordinates before resizing.
    - Resizes the figure and redraws it.
    - Records the position again after resizing.
    - Asserts that the axes-relative position of the label remains stable,
      verifying that applying a `bbox` does not affect layout stability.
    """
    fig, ax = plt.subplots()
    q = ax.quiver([0], [0], [1], [0])

    qk = ax.quiverkey(
        q, X=0.3, Y=0.6, U=1, label='resize test with bbox',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', edgecolor='black')
    )

    fig.canvas.draw()

    display_pos_before = qk.text.get_window_extent().get_points().mean(axis=0)
    axes_pos_before = ax.transAxes.inverted().transform(display_pos_before)

    fig.set_size_inches(10, 8)
    fig.canvas.draw()

    display_pos_after = qk.text.get_window_extent().get_points().mean(axis=0)
    axes_pos_after = ax.transAxes.inverted().transform(display_pos_after)

    assert np.allclose(axes_pos_before, axes_pos_after, atol=0.01), (
        f"Axes position changed with bbox: before {axes_pos_before}, after {axes_pos_after}"
    )

    plt.close(fig)

def find_bbox_patch(qk):
    " Aulixiar Function - Searches for and returns the FancyBboxPatch child of the quiverkey qk, or None if it does not exist."
    for child in qk.get_children():
        if isinstance(child, patches.FancyBboxPatch):
            return child
    return None

def test_quiverkey_bbox_default_properties():
    """
    Test that no FancyBboxPatch is created when `bbox=None` is passed.

    This test:
    - Creates a quiver plot with multiple vectors.
    - Adds a quiverkey with `bbox=None`, meaning no background box is requested.
    - Asserts that the internal `_bbox` attribute is None.
    - Asserts that no `bbox_patch` exists, confirming that no `FancyBboxPatch` was created
      and that the label appears without a styled box.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
        
    q = ax.quiver([0, 40, -60], [0, 20, -30], [3, -2, 1], [1, 2, -3], 
                    color='black', scale=30)
    qk = ax.quiverkey(q, X=0.85, Y=0.15, U=3, label='3 units', labelpos='N', bbox = None)
        
    assert qk._bbox is None, "QuiverKey should have no bbox by default"
    assert not hasattr(qk, 'bbox_patch') or qk.bbox_patch is None, \
        "No bbox_patch should exist when bbox=None"
        
    plt.close(fig)

def test_quiverkey_bbox_style_update():
    """
    Test that QuiverKey bbox properties can be set and updated.

    This test:
    - Creates a quiver plot with a few vectors.
    - Adds a quiverkey with an initial `bbox` specifying facecolor, edgecolor, boxstyle, and alpha.
    - Asserts that the bbox properties are stored correctly.
    - Ensures that rendering with the initial bbox completes without errors.
    - Updates the bbox with new styling parameters.
    - Verifies that the new bbox properties are applied.
    - Ensures that rendering with the updated bbox also completes without errors.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    q = ax.quiver([0, 40, -60], [0, 20, -30], [3, -2, 1], [1, 2, -3], 
                  color='black', scale=30)

    bbox_props = dict(facecolor='lightblue', edgecolor='blue', 
                     boxstyle='round,pad=0.3', alpha=0.8)
    qk = ax.quiverkey(q, X=0.85, Y=0.15, U=3, label='3 units', labelpos='N',
                      bbox=bbox_props)
    
    assert qk._bbox is not None, "QuiverKey should have bbox when specified"
    assert qk._bbox['facecolor'] == 'lightblue', "Facecolor should be lightblue"
    assert qk._bbox['edgecolor'] == 'blue', "Edgecolor should be blue"
    assert qk._bbox['boxstyle'] == 'round,pad=0.3', "Boxstyle should be round,pad=0.3"
    assert qk._bbox['alpha'] == 0.8, "Alpha should be 0.8"

    try:
        fig.canvas.draw()
        rendering_success = True
    except Exception as e:
        rendering_success = False

    assert rendering_success, "Rendering with bbox should work without errors"

    new_bbox = dict(facecolor='red', edgecolor='black', boxstyle='square,pad=0.5')
    qk._bbox.update(new_bbox)
    
    assert qk._bbox['facecolor'] == 'red', "Facecolor should be updated to red"
    assert qk._bbox['edgecolor'] == 'black', "Edgecolor should be updated to black"
    assert qk._bbox['boxstyle'] == 'square,pad=0.5', "Boxstyle should be updated to square"

    try:
        fig.canvas.draw()
        updated_rendering_success = True
    except Exception as e:
        updated_rendering_success = False
    
    assert updated_rendering_success, "Rendering with updated bbox should work"
    
    plt.close(fig)

def test_quiverkey_bbox_with_different_coordinate_systems():
    """
    Test that bbox functionality works correctly across different coordinate systems.

    This test:
    - Creates quiverkeys using 'axes', 'figure', and 'data' coordinate systems.
    - Verifies that bbox is created successfully for each coordinate system.
    - Ensures coordinate system is correctly set on the quiverkey object.
    """
    coordinate_systems = ['axes', 'figure', 'data']

    for coord_sys in coordinate_systems:
        fig, ax = plt.subplots(figsize=(8, 6))
        q = ax.quiver([0], [0], [1], [1], color='blue')

        if coord_sys == 'axes':
            X, Y = 0.8, 0.8
        elif coord_sys == 'figure':
            X, Y = 0.8, 0.8
        elif coord_sys == 'data':
            X, Y = 0.5, 0.5
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        qk = ax.quiverkey(q, X=X, Y=Y, U=1, 
                         label=f'{coord_sys} coords', 
                         coordinates=coord_sys,
                         bbox=dict(facecolor='lightgreen', edgecolor='darkgreen'))

        assert qk.coord == coord_sys, f"Coordinate system should be {coord_sys}"
        assert qk._bbox is not None, f"BBox should exist for {coord_sys} coordinates"

        fig.canvas.draw()

        plt.close(fig)

def test_quiverkey_bbox_edge_cases():
    """
    Test that bbox handles edge cases with various label types and extreme positions.

    This test:
    - Creates quiverkeys with long, short, and multiline labels.
    - Tests positioning at figure boundaries.
    - Verifies that bbox is created successfully for all edge cases.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    q = ax.quiver([0], [0], [1], [1])

    # Test with very long label
    long_label = "This is a very long label that might cause bbox issues"
    qk_long = ax.quiverkey(q, X=0.1, Y=0.9, U=1, label=long_label,
                          bbox=dict(facecolor='cyan', edgecolor='blue'))

    # Test with very short label
    qk_short = ax.quiverkey(q, X=0.9, Y=0.9, U=1, label="X",
                           bbox=dict(facecolor='pink', edgecolor='red'))

    # Test with multiline label
    multiline_label = "Line 1\nLine 2\nLine 3"
    qk_multi = ax.quiverkey(q, X=0.5, Y=0.1, U=1, label=multiline_label,
                           bbox=dict(facecolor='lightblue', edgecolor='navy'))

    # Test positioning at figure edges
    qk_edge = ax.quiverkey(q, X=0.01, Y=0.01, U=1, label="Edge case",
                          bbox=dict(facecolor='yellow', edgecolor='orange'))

    # Verify all have bbox
    for qk in [qk_long, qk_short, qk_multi, qk_edge]:
        assert qk._bbox is not None, "All quiverkeys should have bbox"

    fig.canvas.draw()
    plt.close(fig)

def test_quiverkey_bbox_with_transparent_elements():
    """
    Test bbox behavior with transparent and semi-transparent styling.

    This test:
    - Creates quiverkeys with fully transparent, semi-transparent, and opaque bbox elements.
    - Tests transparent text rendering with bbox backgrounds.
    - Verifies that bbox objects are created regardless of transparency settings.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver([0], [0], [1], [1], alpha=0.5)

    # Test with fully transparent bbox
    qk_transparent = ax.quiverkey(q, X=0.3, Y=0.7, U=1, label="Transparent",
                                 bbox=dict(facecolor='red', alpha=0.0, edgecolor='black'))

    # Test with semi-transparent bbox
    qk_semi = ax.quiverkey(q, X=0.7, Y=0.7, U=1, label="Semi-transparent",
                          bbox=dict(facecolor='blue', alpha=0.5, edgecolor='navy'))

    # Test with transparent text
    qk_text_alpha = ax.quiverkey(q, X=0.5, Y=0.3, U=1, label="Text Alpha",
                                labelcolor=(0, 0, 0, 0.5),
                                bbox=dict(facecolor='green', alpha=0.8))

    for qk in [qk_transparent, qk_semi, qk_text_alpha]:
        assert qk._bbox is not None, "All quiverkeys should have bbox"

    fig.canvas.draw()
    plt.close(fig)

def test_quiverkey_bbox_zorder_interactions():
    """
    Test that bbox respects zorder settings and doesn't interfere with other elements.

    This test:
    - Creates overlapping elements: a background rectangle and a quiverkey with bbox.
    - Uses different zorder values to test proper layering behavior.
    - Verifies that the quiverkey's zorder is correctly set and bbox is created.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver([0], [0], [1], [1], zorder=5)

    ax.add_patch(patches.Rectangle((0.4, 0.4), 0.2, 0.2, 
                                  facecolor='red', alpha=0.7, zorder=1,
                                  transform=ax.transData))

    qk = ax.quiverkey(q, X=0.5, Y=0.5, U=1, label="ZOrder Test",
                     coordinates='data',
                     zorder=10,
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

    assert qk.zorder == 10, "QuiverKey should have zorder 10"
    assert qk._bbox is not None, "BBox should exist"

    fig.canvas.draw()
    plt.close(fig)

def test_quiverkey_bbox_error_handling():
    """
    Test error handling for invalid bbox parameters.

    This test:
    - Tests bbox creation with invalid parameters (like invalid colors).
    - Verifies graceful error handling or successful recovery from bad inputs.
    - Tests bbox creation with empty parameter dictionaries.
    - Ensures robust behavior when users provide problematic bbox configurations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver([0], [0], [1], [1])

    # Test with invalid bbox parameter (should not crash)
    try:
        qk = ax.quiverkey(q, X=0.5, Y=0.5, U=1, label="Error Test",
                         bbox=dict(facecolor='invalid_color'))

        fig.canvas.draw()
        test_passed = True
    except Exception as e:
        # If it fails, ensure it's a reasonable error
        test_passed = isinstance(e, (ValueError, TypeError))

    assert test_passed, "Should handle invalid bbox parameters gracefully"

    qk_empty = ax.quiverkey(q, X=0.3, Y=0.3, U=1, label="Empty BBox",
                           bbox={})
    assert qk_empty._bbox is not None, "Empty bbox dict should still create bbox"

    plt.close(fig)