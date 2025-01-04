import platform

import pytest

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal


def test_indicate_inset_no_args():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='At least one of bounds or inset_ax'):
        ax.indicate_inset()


@check_figures_equal(extensions=["png"])
def test_zoom_inset_update_limits(fig_test, fig_ref):
    # Updating the inset axes limits should also update the indicator #19768
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()

    for ax in ax_ref, ax_test:
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])

    inset_ref = ax_ref.inset_axes([0.6, 0.6, 0.3, 0.3])
    inset_test = ax_test.inset_axes([0.6, 0.6, 0.3, 0.3])

    inset_ref.set_xlim([1, 2])
    inset_ref.set_ylim([3, 4])
    ax_ref.indicate_inset_zoom(inset_ref)

    ax_test.indicate_inset_zoom(inset_test)
    inset_test.set_xlim([1, 2])
    inset_test.set_ylim([3, 4])


def test_inset_indicator_update_styles():
    fig, ax = plt.subplots()
    inset = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
    inset.set_xlim([0.2, 0.4])
    inset.set_ylim([0.2, 0.4])

    indicator = ax.indicate_inset_zoom(
        inset, edgecolor='red', alpha=0.5, linewidth=2, linestyle='solid')

    # Changing the rectangle styles should not affect the connectors.
    indicator.rectangle.set(color='blue', linestyle='dashed', linewidth=42, alpha=0.2)
    for conn in indicator.connectors:
        assert mcolors.same_color(conn.get_edgecolor()[:3], 'red')
        assert conn.get_alpha() == 0.5
        assert conn.get_linestyle() == 'solid'
        assert conn.get_linewidth() == 2

    # Changing the indicator styles should affect both rectangle and connectors.
    indicator.set(color='green', linestyle='dotted', linewidth=7, alpha=0.8)
    assert mcolors.same_color(indicator.rectangle.get_facecolor()[:3], 'green')
    for patch in (*indicator.connectors, indicator.rectangle):
        assert mcolors.same_color(patch.get_edgecolor()[:3], 'green')
        assert patch.get_alpha() == 0.8
        assert patch.get_linestyle() == 'dotted'
        assert patch.get_linewidth() == 7

    indicator.set_edgecolor('purple')
    for patch in (*indicator.connectors, indicator.rectangle):
        assert mcolors.same_color(patch.get_edgecolor()[:3], 'purple')

    # This should also be true if connectors weren't created yet.
    indicator._connectors = []
    indicator.set(color='burlywood', linestyle='dashdot', linewidth=4, alpha=0.4)
    assert mcolors.same_color(indicator.rectangle.get_facecolor()[:3], 'burlywood')
    for patch in (*indicator.connectors, indicator.rectangle):
        assert mcolors.same_color(patch.get_edgecolor()[:3], 'burlywood')
        assert patch.get_alpha() == 0.4
        assert patch.get_linestyle() == 'dashdot'
        assert patch.get_linewidth() == 4

    indicator._connectors = []
    indicator.set_edgecolor('thistle')
    for patch in (*indicator.connectors, indicator.rectangle):
        assert mcolors.same_color(patch.get_edgecolor()[:3], 'thistle')


def test_inset_indicator_zorder():
    fig, ax = plt.subplots()
    rect = [0.2, 0.2, 0.3, 0.4]

    inset = ax.indicate_inset(rect)
    assert inset.get_zorder() == 4.99

    inset = ax.indicate_inset(rect, zorder=42)
    assert inset.get_zorder() == 42


@image_comparison(['zoom_inset_connector_styles.png'], remove_text=True, style='mpl20',
                  tol=0.024 if platform.machine() == 'arm64' else 0)
def test_zoom_inset_connector_styles():
    fig, axs = plt.subplots(2)
    for ax in axs:
        ax.plot([1, 2, 3])

    axs[1].set_xlim(0.5, 1.5)
    indicator = axs[0].indicate_inset_zoom(axs[1], linewidth=5)
    # Make one visible connector a different style
    indicator.connectors[1].set_linestyle('dashed')
    indicator.connectors[1].set_color('blue')


@image_comparison(['zoom_inset_transform.png'], remove_text=True, style='mpl20',
                  tol=0.01)
def test_zoom_inset_transform():
    fig, ax = plt.subplots()

    ax_ins = ax.inset_axes([0.2, 0.2, 0.3, 0.15])
    ax_ins.set_ylim([0.3, 0.6])
    ax_ins.set_xlim([0.5, 0.9])

    tr = mtransforms.Affine2D().rotate_deg(30)
    indicator = ax.indicate_inset_zoom(ax_ins, transform=tr + ax.transData)
    for conn in indicator.connectors:
        conn.set_visible(True)


def test_zoom_inset_external_transform():
    # Smoke test that an external transform that requires an axes (i.e.
    # Cartopy) will work.
    class FussyDataTr:
        def _as_mpl_transform(self, axes=None):
            if axes is None:
                raise ValueError("I am a fussy transform that requires an axes")
            return axes.transData

    fig, ax = plt.subplots()

    ax_ins = ax.inset_axes([0.2, 0.2, 0.3, 0.15])
    ax_ins.set_xlim([0.7, 0.8])
    ax_ins.set_ylim([0.7, 0.8])

    ax.indicate_inset_zoom(ax_ins, transform=FussyDataTr())

    fig.draw_without_rendering()
