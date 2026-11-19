import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison


@image_comparison(['multivar_cbar_locationing.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_multivar_location():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3)
    # -------------------
    locations = ['left', 'right', 'top', 'bottom']
    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.ravel()):
        mim = ax.imshow(data, cmap='2VarAddA')
        fig.colorbar_multivar(mim, location=locations[i])


@image_comparison(['multivar_cbar_sharing.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_multivar_sharing():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3,
            np.arange(12).reshape((3, 4)) % 5)
    # -------------------
    plt.figure()
    ax1 = plt.subplot(211, anchor='NE', aspect='equal')
    plt.imshow(data, cmap='3VarAddA')
    ax2 = plt.subplot(223)
    plt.imshow(data, cmap='3VarAddA')
    ax3 = plt.subplot(224)
    plt.imshow(data, cmap='3VarAddA')

    plt.colorbar_multivar(ax=[ax2, ax3, ax1], location='right')
    plt.colorbar_multivar(ax=[ax2, ax3, ax1], location='left')
    plt.colorbar_multivar(ax=[ax1], location='bottom')


def test_multivar_cbar_remove_from_figure():
    """Test that `remove` restores the position of the parent Axes."""
    fig = plt.figure()
    ax = fig.add_subplot()
    mp = ax.pcolormesh([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                       cmap='2VarAddA',
                       )
    pre_position = ax.get_position()
    cb = fig.colorbar_multivar(mp)
    fig.subplots_adjust()
    cb.remove()
    fig.subplots_adjust()
    post_position = ax.get_position()
    assert (pre_position.get_points() == post_position.get_points()).all()


def test_multivar_cbar_from_colorizer():
    fig, ax = plt.subplots()
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('2VarAddA', norm)
    cb = fig.colorbar_multivar(ca, ax=ax)


def test_multivar_cbar_from_colorizer_cax():
    fig, axes = plt.subplots(1, 3)
    norm = mpl.colors.MultiNorm(['linear', 'linear', 'linear'])
    ca = mpl.colorizer.Colorizer('3VarAddA', norm)
    cb = fig.colorbar_multivar(ca, caxes=axes)


def test_multivar_cbar_set_label_limits():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3,
            np.arange(12).reshape((3, 4)) % 5)
    # -------------------
    fig, ax = plt.subplots(1, 1)
    mim = ax.imshow(data, cmap='3VarAddA')
    cbs = fig.colorbar_multivar(mim)
    assert len(cbs) == 3
    cbs[0].set_label('A')
    assert cbs[0].ax.get_ylabel() == 'A'

    mim.norm.vmin = (-1, -1, -1)
    mim.norm.vmax = (1, 2, 3)
    np.testing.assert_allclose(cbs[0].ax.get_ylim(), (-1, 1))
    np.testing.assert_allclose(cbs[2].ax.get_ylim(), (-1, 3))


def test_wrong_kind_colorbar_multivar():
    fig, ax = plt.subplots(1, 1)
    mp = ax.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                   cmap='2VarAddA',
                   )
    with pytest.raises(ValueError, match='used together with a scalar colormap'):
        fig.colorbar(mp, ax=ax)

    fig, ax = plt.subplots(1, 1)
    mp = ax.imshow([[100, 1], [10, 1]])
    with pytest.raises(ValueError, match='multivariate colorbar can only be used '):
        fig.colorbar_multivar(mp, ax=ax)


@pytest.mark.parametrize('location', ['left', 'right', 'top', 'bottom'])
@pytest.mark.parametrize('n_major, shape', [(-1, (3, 1)), (1, (1, 3)),
                                            (2, (2, 2)), (3, (3, 1))])
def test_colorbar_multivar_n_major_grid(location, n_major, shape):
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3,
            np.arange(12).reshape((3, 4)) % 5)
    fig, ax = plt.subplots()
    mim = ax.imshow(data, cmap='3VarAddA')
    cbs = fig.colorbar_multivar(mim, n_major=n_major, location=location)
    assert len(cbs) == 3
    assert (cbs._colorbar_info['n_major'], cbs._colorbar_info['n_minor']) == shape
    # The colorbars do not overlap each other or the parent Axes.
    bboxes = [ax.get_position()] + [cb.ax.get_position() for cb in cbs]
    for i, bbox_i in enumerate(bboxes):
        for bbox_j in bboxes[i + 1:]:
            assert not bbox_i.overlaps(bbox_j)

    with pytest.raises(ValueError, match="cannot be zero"):
        fig.colorbar_multivar(mim, n_major=0)


@image_comparison(['multivar_cbar_locationing_constrained.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_multivar_location_constrained():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3)
    # -------------------
    locations = ['left', 'right', 'top', 'bottom']
    fig, axes = plt.subplots(2, 2, constrained_layout='constrained')
    for i, ax in enumerate(axes.ravel()):
        mim = ax.imshow(data, cmap='2VarAddA')
        fig.colorbar_multivar(mim, location=locations[i])


@image_comparison(['multivar_cbar_n_major.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_multivar_n_major():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3,
            np.arange(12).reshape((3, 4)) % 5)
    # -------------------
    fig, axes = plt.subplots(4, 3, figsize=(8, 8), constrained_layout='constrained')
    locations = ['left', 'right', 'top', 'bottom']
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            mim = ax.imshow(data, cmap='3VarAddA')
            fig.colorbar_multivar(mim, n_major=j + 1, location=locations[i])

    with pytest.raises(ValueError, match="cannot be zero"):
        fig.colorbar_multivar(mim, n_major=0)
