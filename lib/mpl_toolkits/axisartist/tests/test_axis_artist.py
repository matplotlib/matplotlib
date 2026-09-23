import copy
import gc
import weakref

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.font_manager import FontManager, FontProperties
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.text import Text
from matplotlib.transforms import Affine2D

from mpl_toolkits.axisartist import (AxisArtistHelperRectlinear, GridHelperCurveLinear,
                                     HostAxes)
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,
                                                 LabelBase, Ticks, TickLabels)


@pytest.mark.parametrize("coordinate, side", [("x", "bottom"), ("y", "left")])
@pytest.mark.parametrize(
    "property_name", ["fontsize", "size", "fontdict", "fontproperties"])
@check_figures_equal(extensions=["png", "pdf", "svg"])
def test_label_fontsize_from_axes(fig_test, fig_ref, coordinate, side, property_name):
    for fig in [fig_test, fig_ref]:
        ax = fig.add_subplot(axes_class=HostAxes)
        ax.plot([0, 1], [0, 1])
        for size in [20, 14]:
            value = ({"fontsize": size} if property_name == "fontdict" else
                     FontProperties(size=size) if property_name == "fontproperties" else
                     size)
            getattr(ax, f"set_{coordinate}label")(
                "Axis label", **{property_name: value})
            if fig is fig_ref:
                ax.axis[side].label.set_fontsize(size)
            fig.canvas.draw()


@pytest.mark.parametrize("method, value, expected", [
    ("set_fontsize", 20, 20),
    ("set_fontsize", 30, 30), ("set_size", 30, 30),
    ("set_fontproperties", FontProperties(size=30), 30),
    ("set_font", ":size=30", 30),
])
def test_label_fontsize_override(method, value, expected):
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_ylabel("Axis label", fontsize=20)
    label = ax.axis["left"].label
    getattr(label, method)(value)
    ax.set_ylabel("New label", fontsize=14)
    fig.canvas.draw()
    assert label.get_fontsize() == expected
    assert label.get_fontproperties().get_size_in_points() == expected
    assert ax.yaxis.label.get_fontsize() == 14


def test_label_fontproperties_mutable():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_ylabel("Axis label", fontsize=20)
    label = ax.axis["left"].label
    properties = label.get_fontproperties()
    properties.set_weight("bold")
    properties.set_size(30)
    ax.set_ylabel("New label", fontsize=14)
    fig.canvas.draw()
    assert label.get_fontproperties() is properties
    assert label.get_fontweight() == "bold"
    assert label.get_fontsize() == 30


def test_label_fontproperties_update_from():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_ylabel("Axis label", fontsize=20)
    label = ax.axis["left"].label
    text = Text()
    text.update_from(label)
    assert text.get_fontsize() == 20
    label.update_from(Text(fontsize=30))
    ax.set_ylabel("New label", fontsize=14)
    assert label.get_fontsize() == 30


def test_label_fontproperties_copy():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_ylabel("Axis label", fontsize=20)
    properties = ax.axis["left"].label.get_fontproperties()
    snapshot = copy.copy(properties)
    assert hash(properties) == hash(snapshot)
    ax.set_ylabel("New label", fontsize=14)
    assert properties.get_size_in_points() == 14
    assert snapshot.get_size_in_points() == 20
    assert properties != snapshot


def test_label_font_lookup_does_not_keep_figure_alive():
    manager = FontManager()
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_xlabel("Axis label", fontsize=20)
    manager.findfont(ax.axis["bottom"].label.get_fontproperties())
    reference = weakref.ref(fig)
    plt.close(fig)
    del fig, ax
    gc.collect()
    assert reference() is None


def test_label_fontsize_bbox():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=HostAxes)
    ax.set_xlabel("Axis label", fontsize=20)
    label = ax.axis["bottom"].label
    label.set_bbox({"facecolor": "white"})
    fig.canvas.draw()
    patch = label.get_bbox_patch()
    size = patch.get_width(), patch.get_height()
    label.update_bbox_position_size(fig.canvas.get_renderer())
    assert (patch.get_width(), patch.get_height()) == size


@image_comparison(['axis_artist_ticks.png'], style='default')
def test_ticks():
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    locs_angles = [((i / 10, 0.0), i * 30) for i in range(-1, 12)]

    ticks_in = Ticks(ticksize=10, axis=ax.xaxis)
    ticks_in.set_locs_angles(locs_angles)
    ax.add_artist(ticks_in)

    ticks_out = Ticks(ticksize=10, tick_direction="out", color='C3', axis=ax.xaxis)
    ticks_out.set_locs_angles(locs_angles)
    ax.add_artist(ticks_out)


@image_comparison(['axis_artist_labelbase.png'], style='default')
def test_labelbase():
    fig, ax = plt.subplots()

    ax.plot([0.5], [0.5], "o")

    label = LabelBase(0.5, 0.5, "Test")
    label._ref_angle = -90
    label._offset_radius = 50
    label.set_rotation(-90)
    label.set(ha="center", va="top")
    ax.add_artist(label)


@image_comparison(['axis_artist_ticklabels.png'], style='default')
def test_ticklabels():
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.plot([0.2, 0.4], [0.5, 0.5], "o")

    ticks = Ticks(ticksize=10, axis=ax.xaxis)
    ax.add_artist(ticks)
    locs_angles_labels = [((0.2, 0.5), -90, "0.2"),
                          ((0.4, 0.5), -120, "0.4")]
    tick_locs_angles = [(xy, a + 180) for xy, a, l in locs_angles_labels]
    ticks.set_locs_angles(tick_locs_angles)

    ticklabels = TickLabels(axis_direction="left")
    ticklabels._locs_angles_labels = locs_angles_labels
    ticklabels.set_pad(10)
    ax.add_artist(ticklabels)

    ax.plot([0.5], [0.5], "s")
    axislabel = AxisLabel(0.5, 0.5, "Test")
    axislabel._offset_radius = 20
    axislabel._ref_angle = 0
    axislabel.set_axis_direction("bottom")
    ax.add_artist(axislabel)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


@image_comparison(['axis_artist.png'], style='default')
def test_axis_artist():
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    for loc in ('left', 'right', 'bottom'):
        helper = AxisArtistHelperRectlinear.Fixed(ax, loc=loc)
        axisline = AxisArtist(ax, helper, offset=None, axis_direction=loc)
        axisline.major_ticks.set_tick_direction({
            "left": "in", "right": "out", "bottom": "inout",
        }[loc])
        ax.add_artist(axisline)

    # Settings for bottom AxisArtist.
    axisline.set_label("TTT")
    axisline.label.set_pad(5)

    ax.set_ylabel("Test")


@mpl.style.context('default')
def test_axisartist_tightbbox():
    fig = plt.figure()
    tr = Affine2D().scale(np.pi / 180., 1.) + PolarAxes.PolarTransform()
    grid_helper = GridHelperCurveLinear(tr)
    ax = fig.add_subplot(axes_class=HostAxes, grid_helper=grid_helper)
    ax.axis["lon"] = ax.new_floating_axis(1, 9)

    ax.set_xlim(-5, 12)
    ax.set_ylim(-5, 10)

    ax.axis['lon'].major_ticklabels.set_visible(False)

    # Since the labels are invisible and the lines are clipped to the axes,
    # the axis's tight bbox should be contained in the axes box.
    renderer = fig._get_renderer()
    tight_points = ax.axis['lon'].get_tightbbox(renderer).get_points()
    for point in tight_points:
        assert ax.bbox.contains(*point)
