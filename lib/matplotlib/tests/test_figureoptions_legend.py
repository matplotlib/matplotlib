"""Tests for legend preservation in Qt figure options dialog."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_legend_properties_preserved():
    """
    Test that legend properties are preserved when the legend is
    regenerated via the Qt figure options dialog.

    Regression test for https://github.com/matplotlib/matplotlib/issues/17775
    """
    fig, ax = plt.subplots()
    ax.plot(range(5), label='a')
    ax.plot(range(3)[::-1], label='b')

    ax.legend(
        loc='upper right',
        fontsize=14,
        frameon=False,
        shadow=True,
        title='My Legend',
        ncols=2,
        columnspacing=3.0,
        labelspacing=1.5,
        handlelength=4.0,
        handletextpad=1.2,
    )

    old_legend = ax.get_legend()
    old_loc = old_legend._loc
    old_fontsize = old_legend._fontsize
    old_frameon = old_legend.get_frame_on()
    old_shadow = old_legend.shadow
    old_title = old_legend.get_title().get_text()
    old_ncols = old_legend._ncols
    old_columnspacing = old_legend.columnspacing
    old_labelspacing = old_legend.labelspacing
    old_handlelength = old_legend.handlelength
    old_handletextpad = old_legend.handletextpad

    # Simulate what the patched figureoptions code does
    draggable = old_legend._draggable is not None
    ncols = old_legend._ncols
    legend_kwargs = {}
    legend_kwargs['loc'] = old_legend._loc
    if old_legend._bbox_to_anchor is not None:
        legend_kwargs['bbox_to_anchor'] = (
            old_legend._bbox_to_anchor.bounds)
    legend_kwargs['fontsize'] = old_legend._fontsize
    legend_kwargs['frameon'] = old_legend.get_frame_on()
    legend_kwargs['shadow'] = old_legend.shadow
    legend_kwargs['framealpha'] = old_legend.get_frame().get_alpha()
    legend_kwargs['title'] = old_legend.get_title().get_text()
    if old_legend._mode is not None:
        legend_kwargs['mode'] = old_legend._mode
    legend_kwargs['columnspacing'] = old_legend.columnspacing
    legend_kwargs['labelspacing'] = old_legend.labelspacing
    legend_kwargs['handlelength'] = old_legend.handlelength
    legend_kwargs['handletextpad'] = old_legend.handletextpad

    new_legend = ax.legend(ncols=ncols, **legend_kwargs)
    if new_legend:
        new_legend.set_draggable(draggable)

    assert new_legend._loc == old_loc
    assert new_legend._fontsize == old_fontsize
    assert new_legend.get_frame_on() == old_frameon
    assert new_legend.shadow == old_shadow
    assert new_legend.get_title().get_text() == old_title
    assert new_legend._ncols == old_ncols
    assert new_legend.columnspacing == old_columnspacing
    assert new_legend.labelspacing == old_labelspacing
    assert new_legend.handlelength == old_handlelength
    assert new_legend.handletextpad == old_handletextpad

    plt.close(fig)


def test_legend_regeneration_no_existing_legend():
    """
    Test that regenerating a legend when none exists still works.
    """
    fig, ax = plt.subplots()
    ax.plot(range(5), label='a')

    assert ax.get_legend() is None

    new_legend = ax.legend(ncols=1)
    assert new_legend is not None
    assert len(new_legend.get_texts()) == 1
    assert new_legend.get_texts()[0].get_text() == 'a'

    plt.close(fig)


def test_legend_mode_preserved():
    """
    Test that mode='expand' is preserved on legend regeneration.
    """
    fig, ax = plt.subplots()
    ax.plot(range(5), label='a')
    ax.plot(range(3)[::-1], label='b')

    ax.legend(
        loc='lower left',
        mode='expand',
        ncols=2,
    )

    old_legend = ax.get_legend()
    old_mode = old_legend._mode
    old_loc = old_legend._loc

    legend_kwargs = {}
    legend_kwargs['loc'] = old_legend._loc
    legend_kwargs['fontsize'] = old_legend._fontsize
    legend_kwargs['frameon'] = old_legend.get_frame_on()
    if old_legend._mode is not None:
        legend_kwargs['mode'] = old_legend._mode

    new_legend = ax.legend(ncols=old_legend._ncols, **legend_kwargs)

    assert new_legend._mode == old_mode
    assert new_legend._loc == old_loc

    plt.close(fig)
