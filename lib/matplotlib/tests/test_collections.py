"""
Tests specific to the collections module.
"""

from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_array_equal

import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.transforms as mtransforms
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

    coll = EventCollection(positions,
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
    splt.add_collection(coll)
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
    return splt, coll, props


@image_comparison(baseline_images=['EventCollection_plot__default'])
def test__EventCollection__get_segments():
    '''
    check to make sure the default segments have the correct coordinates
    '''
    _, coll, props = generate_EventCollection_plot()
    check_segments(coll,
                   props['positions'],
                   props['linelength'],
                   props['lineoffset'],
                   props['orientation'])


@cleanup
def test__EventCollection__get_positions():
    '''
    check to make sure the default positions match the input positions
    '''
    _, coll, props = generate_EventCollection_plot()
    np.testing.assert_array_equal(props['positions'], coll.get_positions())


@cleanup
def test__EventCollection__get_orientation():
    '''
    check to make sure the default orientation matches the input
    orientation
    '''
    _, coll, props = generate_EventCollection_plot()
    assert_equal(props['orientation'], coll.get_orientation())


@cleanup
def test__EventCollection__is_horizontal():
    '''
    check to make sure the default orientation matches the input
    orientation
    '''
    _, coll, _ = generate_EventCollection_plot()
    assert_equal(True, coll.is_horizontal())


@cleanup
def test__EventCollection__get_linelength():
    '''
    check to make sure the default linelength matches the input linelength
    '''
    _, coll, props = generate_EventCollection_plot()
    assert_equal(props['linelength'], coll.get_linelength())


@cleanup
def test__EventCollection__get_lineoffset():
    '''
    check to make sure the default lineoffset matches the input lineoffset
    '''
    _, coll, props = generate_EventCollection_plot()
    assert_equal(props['lineoffset'], coll.get_lineoffset())


@cleanup
def test__EventCollection__get_linestyle():
    '''
    check to make sure the default linestyle matches the input linestyle
    '''
    _, coll, _ = generate_EventCollection_plot()
    assert_equal(coll.get_linestyle(), [(None, None)])


@cleanup
def test__EventCollection__get_color():
    '''
    check to make sure the default color matches the input color
    '''
    _, coll, props = generate_EventCollection_plot()
    np.testing.assert_array_equal(props['color'], coll.get_color())
    check_allprop_array(coll.get_colors(), props['color'])


@image_comparison(baseline_images=['EventCollection_plot__set_positions'])
def test__EventCollection__set_positions():
    '''
    check to make sure set_positions works properly
    '''
    splt, coll, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'], props['extra_positions']])
    coll.set_positions(new_positions)
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    check_segments(coll, new_positions,
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
    splt, coll, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][0]])
    coll.add_positions(props['extra_positions'][0])
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][2]])
    coll.append_positions(props['extra_positions'][2])
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'],
                               props['extra_positions'][1:]])
    coll.extend_positions(props['extra_positions'][1:])
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_orientation = 'vertical'
    coll.switch_orientation()
    assert_equal(new_orientation, coll.get_orientation())
    assert_equal(False, coll.is_horizontal())
    new_positions = coll.get_positions()
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    coll.switch_orientation()
    coll.switch_orientation()
    new_positions = coll.get_positions()
    assert_equal(props['orientation'], coll.get_orientation())
    assert_equal(True, coll.is_horizontal())
    np.testing.assert_array_equal(props['positions'], new_positions)
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_orientation = 'vertical'
    coll.set_orientation(new_orientation)
    assert_equal(new_orientation, coll.get_orientation())
    assert_equal(False, coll.is_horizontal())
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_linelength = 15
    coll.set_linelength(new_linelength)
    assert_equal(new_linelength, coll.get_linelength())
    check_segments(coll,
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
    splt, coll, props = generate_EventCollection_plot()
    new_lineoffset = -5.
    coll.set_lineoffset(new_lineoffset)
    assert_equal(new_lineoffset, coll.get_lineoffset())
    check_segments(coll,
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
    splt, coll, _ = generate_EventCollection_plot()
    new_linestyle = 'dashed'
    coll.set_linestyle(new_linestyle)
    assert_equal(coll.get_linestyle(), [(0, (6.0, 6.0))])
    splt.set_title('EventCollection: set_linestyle')


@image_comparison(baseline_images=['EventCollection_plot__set_linewidth'])
def test__EventCollection__set_linewidth():
    '''
    check to make sure set_linestyle works properly
    '''
    splt, coll, _ = generate_EventCollection_plot()
    new_linewidth = 5
    coll.set_linewidth(new_linewidth)
    assert_equal(coll.get_linewidth(), new_linewidth)
    splt.set_title('EventCollection: set_linewidth')


@image_comparison(baseline_images=['EventCollection_plot__set_color'])
def test__EventCollection__set_color():
    '''
    check to make sure set_color works properly
    '''
    splt, coll, _ = generate_EventCollection_plot()
    new_color = np.array([0, 1, 1, 1])
    coll.set_color(new_color)
    np.testing.assert_array_equal(new_color, coll.get_color())
    check_allprop_array(coll.get_colors(), new_color)
    splt.set_title('EventCollection: set_color')


def check_segments(coll, positions, linelength, lineoffset, orientation):
    '''
    check to make sure all values in the segment are correct, given a
    particular set of inputs

    note: this is not a test, it is used by tests
    '''
    segments = coll.get_segments()
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
        np.testing.assert_array_equal(value, target)


def test_null_collection_datalim():
    col = mcollections.PathCollection([])
    col_data_lim = col.get_datalim(mtransforms.IdentityTransform())
    assert_array_equal(col_data_lim.get_points(),
                       mtransforms.Bbox([[0, 0], [0, 0]]).get_points())


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
