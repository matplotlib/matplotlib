import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

import matplotlib.transforms as mtransforms

from mpl_toolkits import axisartist
from mpl_toolkits.axisartist import AxisArtistHelperRectlinear
from mpl_toolkits.axisartist.axis_artist import (AxisArtist, AxisLabel,
                                                 LabelBase, Ticks, TickLabels)
from mpl_toolkits.axisartist.grid_helper_curvelinear import (
    GridHelperCurveLinear)

from matplotlib._api.deprecation import MatplotlibDeprecationWarning


@image_comparison(['axis_artist_ticks.png'], style='default')
def test_ticks():
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    locs_angles = [((i / 10, 0.0), i * 30) for i in range(-1, 12)]

    ticks_in = Ticks(ticksize=10, tickdir="in", axis=ax.xaxis)
    ticks_in.set_locs_angles(locs_angles)
    ax.add_artist(ticks_in)

    ticks_out = Ticks(ticksize=10, tickdir="out", color='C3', axis=ax.xaxis)
    ticks_out.set_locs_angles(locs_angles)
    ax.add_artist(ticks_out)


@image_comparison(['axis_artist_labelbase.png'], style='default')
def test_labelbase():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

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
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.plot([0.2, 0.4], [0.5, 0.5], "o")

    ticks = Ticks(ticksize=10, tickdir="in", axis=ax.xaxis)
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
    # Remove these lines when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6
    plt.rcParams.update({"xtick.direction": "in", "ytick.direction": "in"})

    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    for loc in ('left', 'right', 'bottom'):
        _helper = AxisArtistHelperRectlinear.Fixed(ax, loc=loc)
        axisline = AxisArtist(ax, _helper, offset=None, axis_direction=loc)
        ax.add_artist(axisline)

    # Settings for bottom AxisArtist.
    axisline.set_label("TTT")
    axisline.major_ticks.set_tick_out(False)
    axisline.label.set_pad(5)

    ax.set_ylabel("Test")


def test_tickout_kwargs(recwarn):
    """
    test that 'singular' versions of LineCollection props raise an
    MatplotlibDeprecationWarning rather than overriding the 'plural' versions
    (e.g., to prevent 'color' from overriding 'colors', see issue #4297)
    """

    Ticks(1)
    Ticks(1, False)
    Ticks(1, False)
    Ticks(1, tick_out=True)
    Ticks(1, tick_out=False)

    assert issubclass(recwarn[0].category, UserWarning)
    # May need to check the message

    assert all(issubclass(wi.category, MatplotlibDeprecationWarning)
               for wi in recwarn[1:])


def _setup_axes_for_axis_artist_dir(fig, pos):
    ax = fig.add_subplot(pos, axes_class=axisartist.Axes)

    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 1])

    ax.axis[:].set_visible(False)

    ax.axis["x"] = ax.new_floating_axis(1, 0.5)

    return ax


@image_comparison(['axis_artist_dir.png'], style='default')
def test_axis_artist_dir():
    plt.rcParams['text.kerning_factor'] = 6

    fig = plt.figure(figsize=(4, 5), num=1)
    fig.clf()

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig)
    gsi = iter(gs)

    for td in ["in", "out", "inout"]:
        for ld in "+-":
            for tld in "+-":
                ax = _setup_axes_for_axis_artist_dir(fig, next(gsi))
                axis = ax.axis["x"]
                axis.major_ticks.set_ticksize(8)
                axis.label.set_text("Label")
                axis.toggle(ticklabels=True)
                axis.set_axislabel_direction(ld)
                axis.set_ticklabel_direction(tld)
                axis.major_ticks.set_tickdir(td)


def _setup_axes_tick_orientation(fig, pos):
    tr = mtransforms.Affine2D().skew_deg(0, 30)

    grid_helper = GridHelperCurveLinear(tr)

    ax = fig.add_subplot(pos, axes_class=axisartist.Axes,
                         grid_helper=grid_helper)

    ax.set_xlim(0, 0.2)

    ax.axis[:].set_visible(False)

    ax.axis["test"] = ax.new_fixed_axis("left")
    ax.axis["test"].toggle(ticklabels=False)

    ax.set_aspect(1)
    ax.grid(True, axis="y", color="0.8")
    return ax


@image_comparison(['axis_artist_tick_orientation.png'], style='default')
def test_axis_artist_tick_orientation():
    plt.rcParams['text.kerning_factor'] = 6

    fig = plt.figure(figsize=(4, 5))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig)
    gsi = iter(gs)

    for td in ["in", "out", "inout"]:
        for to in ["auto", "normal", "parallel"]:
            ax = _setup_axes_tick_orientation(fig, next(gsi))
            axis = ax.axis["test"]
            axis.major_ticks.set_ticksize(8)
            axis.major_ticks.set_tickdir(td)
            axis.major_ticks.set_tick_orientation(to)
            axis.label.set_text(f"{td}-{to}")
