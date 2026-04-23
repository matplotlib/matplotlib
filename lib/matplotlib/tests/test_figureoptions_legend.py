"""Tests for legend property preservation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_get_properties_basic():
    """
    Test that Legend._get_properties() returns the correct properties.
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

    legend = ax.get_legend()
    props = legend._get_properties()

    assert props['loc'] == legend._loc
    assert props['fontsize'] == 14
    assert props['frameon'] is False
    assert props['shadow'] is True
    assert props['title'] == 'My Legend'
    assert props['columnspacing'] == 3.0
    assert props['labelspacing'] == 1.5
    assert props['handlelength'] == 4.0
    assert props['handletextpad'] == 1.2
    assert 'bbox_to_anchor' not in props
    assert 'mode' not in props

    plt.close(fig)


def test_get_properties_with_mode_and_bbox():
    """
    Test that _get_properties() includes mode and bbox_to_anchor
    when they are set.
    """
    fig, ax = plt.subplots()
    ax.plot(range(5), label='a')

    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc='lower left',
        mode='expand',
        ncols=2,
    )

    legend = ax.get_legend()
    props = legend._get_properties()

    assert props['mode'] == 'expand'
    assert 'bbox_to_anchor' in props

    plt.close(fig)


def test_get_properties_roundtrip():
    """
    Test that properties from _get_properties() can be used to
    recreate a legend with the same settings.
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
    props = old_legend._get_properties()
    ncols = old_legend._ncols

    # Recreate legend using extracted properties
    new_legend = ax.legend(ncols=ncols, **props)

    assert new_legend._fontsize == 14
    assert new_legend.get_frame_on() is False
    assert new_legend.shadow is True
    assert new_legend.get_title().get_text() == 'My Legend'
    assert new_legend._ncols == 2
    assert new_legend.columnspacing == 3.0
    assert new_legend.labelspacing == 1.5
    assert new_legend.handlelength == 4.0
    assert new_legend.handletextpad == 1.2

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
