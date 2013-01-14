"""
Tests specific to the collections module.
"""

from nose.tools import assert_equal
from numpy.testing import assert_array_equal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from matplotlib.testing.decorators import cleanup, image_comparison


def generate_EventCollection_plot():
    '''
    generate the initial collection and plot it
    '''
    positions = np.array([0., 1., 2., 3., 5., 8., 13., 21.])
    extra_positions = np.array([34., 55., 89.])
    orientation = 'horizontal'
    lineoffset = 1
    linelength = .5
    linewidth = 2
    color = [1, 0, 0, 1]
    linestyle = 'solid'
    antialiased = True

    event_col = EventCollection(positions,
                                orientation=orientation,
                                lineoffset=lineoffset,
                                linelength=linelength,
                                linewidth=linewidth,
                                color=color,
                                linestyle=linestyle,
                                antialiased=antialiased
                                )

    fig = plt.figure()
    splt = fig.add_subplot(1, 1, 1)
    splt.add_collection(event_col)
    splt.set_title('EventCollection: default')
    props = {'positions': positions,
             'extra_positions': extra_positions,
             'orientation': orientation,
             'lineoffset': lineoffset,
             'linelength': linelength,
             'linewidth': linewidth,
             'color': color,
             'linestyle': linestyle,
             'antialiased': antialiased
             }
    splt.set_xlim(-1, 22)
    splt.set_ylim(0, 2)
    return splt, event_col, props


@image_comparison(baseline_images=['EventCollection_plot__default'])
def test__EventCollection__get_segments():
    '''
    check to make sure the default segments have the correct coordinates
    '''
    _, event_col, props = generate_EventCollection_plot()
    check_segments(event_col,
                   props['positions'],
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])


@cleanup
def test__EventCollection__get_positions():
    '''
    check to make sure the default positions match the input positions
    '''
    _, event_col, props = generate_EventCollection_plot()
    assert_array_equal(props['positions'], event_col.get_positions())


@cleanup
def test__EventCollection__get_orientation():
    '''
    check to make sure the default orientation matches the input
    orientation
    '''
    _, event_col, props = generate_EventCollection_plot()
    assert_equal(props['orientation'], event_col.get_orientation())


@cleanup
def test__EventCollection__is_horizontal():
    '''
    check to make sure the default orientation matches the input
    orientation
    '''
    _, event_col, _ = generate_EventCollection_plot()
    assert_equal(True, event_col.is_horizontal())


@cleanup
def test__EventCollection__get_linelength():
    '''
    check to make sure the default linelength matches the input linelength
    '''
    _, event_col, props = generate_EventCollection_plot()
    assert_equal(props['linelength'], event_col.get_linelength())


@cleanup
def test__EventCollection__get_lineoffset():
    '''
    check to make sure the default lineoffset matches the input lineoffset
    '''
    _, event_col, props = generate_EventCollection_plot()
    assert_equal(props['lineoffset'], event_col.get_lineoffset())


@cleanup
def test__EventCollection__get_linestyle():
    '''
    check to make sure the default linestyle matches the input linestyle
    '''
    _, event_col, _ = generate_EventCollection_plot()
    assert_equal(event_col.get_linestyle(), [(None, None)])


@cleanup
def test__EventCollection__get_color():
    '''
    check to make sure the default color matches the input color
    '''
    _, event_col, props = generate_EventCollection_plot()
    assert_array_equal(props['color'], event_col.get_color())
    check_allprop_array(event_col.get_colors(), props['color'])


@image_comparison(baseline_images=['EventCollection_plot__set_positions'])
def test__EventCollection__set_positions():
    '''
    check to make sure set_positions works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'], props['extra_positions']])
    event_col.set_positions(new_positions)
    assert_array_equal(new_positions, event_col.get_positions())
    check_segments(event_col, new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: set_positions')
    splt.set_xlim(-1, 90)


@image_comparison(baseline_images=['EventCollection_plot__add_positions'])
def test__EventCollection__add_positions():
    '''
    check to make sure add_positions works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][0]])
    event_col.add_positions(props['extra_positions'][0])
    assert_array_equal(new_positions, event_col.get_positions())
    check_segments(event_col,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: add_positions')
    splt.set_xlim(-1, 35)


@image_comparison(baseline_images=['EventCollection_plot__append_positions'])
def test__EventCollection__append_positions():
    '''
    check to make sure append_positions works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][2]])
    event_col.append_positions(props['extra_positions'][2])
    assert_array_equal(new_positions, event_col.get_positions())
    check_segments(event_col,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: append_positions')
    splt.set_xlim(-1, 90)


@image_comparison(baseline_images=['EventCollection_plot__extend_positions'])
def test__EventCollection__extend_positions():
    '''
    check to make sure extend_positions works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][1:]])
    event_col.extend_positions(props['extra_positions'][1:])
    assert_array_equal(new_positions, event_col.get_positions())
    check_segments(event_col,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: extend_positions')
    splt.set_xlim(-1, 90)


@image_comparison(baseline_images=['EventCollection_plot__switch_orientation'])
def test__EventCollection__switch_orientation():
    '''
    check to make sure switch_orientation works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_orientation = 'vertical'
    event_col.switch_orientation()
    assert_equal(new_orientation, event_col.get_orientation())
    assert_equal(False, event_col.is_horizontal())
    new_positions = event_col.get_positions()
    check_segments(event_col,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'], new_orientation)
    splt.set_title('EventCollection: switch_orientation')
    splt.set_ylim(-1, 22)
    splt.set_xlim(0, 2)


@image_comparison(baseline_images=
                  ['EventCollection_plot__switch_orientation__2x'])
def test__EventCollection__switch_orientation_2x():
    '''
    check to make sure calling switch_orientation twice sets the
    orientation back to the default
    '''
    splt, event_col, props = generate_EventCollection_plot()
    event_col.switch_orientation()
    event_col.switch_orientation()
    new_positions = event_col.get_positions()
    assert_equal(props['orientation'], event_col.get_orientation())
    assert_equal(True, event_col.is_horizontal())
    assert_array_equal(props['positions'], new_positions)
    check_segments(event_col,
                   new_positions,
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: switch_orientation 2x')


@image_comparison(baseline_images=['EventCollection_plot__set_orientation'])
def test__EventCollection__set_orientation():
    '''
    check to make sure set_orientation works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_orientation = 'vertical'
    event_col.set_orientation(new_orientation)
    assert_equal(new_orientation, event_col.get_orientation())
    assert_equal(False, event_col.is_horizontal())
    check_segments(event_col,
                   props['positions'],
                   props['linelength'],
                   props['lineoffset'],
                   new_orientation)
    splt.set_title('EventCollection: set_orientation')
    splt.set_ylim(-1, 22)
    splt.set_xlim(0, 2)


@image_comparison(baseline_images=['EventCollection_plot__set_linelength'])
def test__EventCollection__set_linelength():
    '''
    check to make sure set_linelength works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_linelength = 15
    event_col.set_linelength(new_linelength)
    assert_equal(new_linelength, event_col.get_linelength())
    check_segments(event_col,
                   props['positions'],
                   new_linelength,
                   props['lineoffset'],
                   props['orientation'])
    splt.set_title('EventCollection: set_linelength')
    splt.set_ylim(-20, 20)


@image_comparison(baseline_images=['EventCollection_plot__set_lineoffset'])
def test__EventCollection__set_lineoffset():
    '''
    check to make sure set_lineoffset works properly
    '''
    splt, event_col, props = generate_EventCollection_plot()
    new_lineoffset = -5.
    event_col.set_lineoffset(new_lineoffset)
    assert_equal(new_lineoffset, event_col.get_lineoffset())
    check_segments(event_col,
                   props['positions'],
                   props['linelength'],
                   new_lineoffset,
                   props['orientation'])
    splt.set_title('EventCollection: set_lineoffset')
    splt.set_ylim(-6, -4)


@image_comparison(baseline_images=['EventCollection_plot__set_linestyle'])
def test__EventCollection__set_linestyle():
    '''
    check to make sure set_linestyle works properly
    '''
    splt, event_col, _ = generate_EventCollection_plot()
    new_linestyle = 'dashed'
    event_col.set_linestyle(new_linestyle)
    assert_equal(event_col.get_linestyle(), [(0, (6.0, 6.0))])
    splt.set_title('EventCollection: set_linestyle')


@image_comparison(baseline_images=['EventCollection_plot__set_linewidth'])
def test__EventCollection__set_linewidth():
    '''
    check to make sure set_linestyle works properly
    '''
    splt, event_col, _ = generate_EventCollection_plot()
    new_linewidth = 5
    event_col.set_linewidth(new_linewidth)
    assert_equal(event_col.get_linewidth(), new_linewidth)
    splt.set_title('EventCollection: set_linewidth')


@image_comparison(baseline_images=['EventCollection_plot__set_color'])
def test__EventCollection__set_color():
    '''
    check to make sure set_color works properly
    '''
    splt, event_col, _ = generate_EventCollection_plot()
    new_color = np.array([0, 1, 1, 1])
    event_col.set_color(new_color)
    assert_array_equal(new_color, event_col.get_color())
    check_allprop_array(event_col.get_colors(), new_color)
    splt.set_title('EventCollection: set_color')


def check_segments(event_col, positions, linelength, lineoffset, orientation):
    '''
    check to make sure all values in the segment are correct, given a
    particular set of inputs

    note: this is not a test, it is used by tests
    '''
    segments = event_col.get_segments()
    if (orientation.lower() == 'horizontal'
            or orientation.lower() == 'none' or orientation is None):
        # if horizontal, the position in is in the y-axis
        pos1 = 1
        pos2 = 0
    elif orientation.lower() == 'vertical':
        # if vertical, the position in is in the x-axis
        pos1 = 0
        pos2 = 1
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    # test to make sure each segment is correct
    for i, segment in enumerate(segments):
        assert_equal(segment[0, pos1], lineoffset + linelength / 2.)
        assert_equal(segment[1, pos1], lineoffset - linelength / 2.)
        assert_equal(segment[0, pos2], positions[i])
        assert_equal(segment[1, pos2], positions[i])


def check_allprop(values, target):
    '''
    check to make sure all values match the given target

    note: this is not a test, it is used by tests
    '''
    for value in values:
        assert_equal(value, target)


def check_allprop_array(values, target):
    '''
    check to make sure all values match the given target if arrays

    note: this is not a test, it is used by tests
    '''
    for value in values:
        assert_array_equal(value, target)

if __name__ == '_main_':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
