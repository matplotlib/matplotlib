from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import re

import numpy as np
import pytest

from mpl_toolkits.axisartist.angle_helper import (
    FormatterDMS, select_step, select_step360)


DMS_RE = re.compile(
    r'''\$  # Mathtext
        (
            # The sign sometimes appears on a 0 when a fraction is shown.
            # Check later that there's only one.
            (?P<degree_sign>-)?
            (?P<degree>[0-9.]+)  # Degrees value
            \^{\\circ}  # Degree symbol
        )?
        (
            (?(degree)\\,)  # Separator if degrees are also visible.
            (?P<minute_sign>-)?
            (?P<minute>[0-9.]+)  # Minutes value
            \^{\\prime}  # Minute symbol
        )?
        (
            (?(minute)\\,)  # Separator if minutes are also visible.
            (?P<second_sign>-)?
            (?P<second>[0-9.]+)  # Seconds value
            \^{\\prime\\prime}  # Second symbol
        )?
        \$  # Mathtext
    ''',
    re.VERBOSE
)


def dms2float(degrees, minutes=0, seconds=0):
    return degrees + minutes / 60.0 + seconds / 3600.0


@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [
    ((-180, 180, 10), {'hour': False}, np.arange(-180, 181, 30), 1.0),
    ((-12, 12, 10), {'hour': True}, np.arange(-12, 13, 2), 1.0)
])
def test_select_step(args, kwargs, expected_levels, expected_factor):
    levels, n, factor = select_step(*args, **kwargs)

    assert n == len(levels)
    np.testing.assert_array_equal(levels, expected_levels)
    assert factor == expected_factor


@pytest.mark.parametrize('args, kwargs, expected_levels, expected_factor', [
    ((dms2float(20, 21.2), dms2float(21, 33.3), 5), {},
     np.arange(1215, 1306, 15), 60.0),
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {},
     np.arange(73820, 73835, 2), 3600.0),
    ((dms2float(20, 21.2), dms2float(20, 53.3), 5), {},
     np.arange(1220, 1256, 5), 60.0),
    ((21.2, 33.3, 5), {},
     np.arange(20, 35, 2), 1.0),
    ((dms2float(20, 21.2), dms2float(21, 33.3), 5), {},
     np.arange(1215, 1306, 15), 60.0),
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=33.3), 5), {},
     np.arange(73820, 73835, 2), 3600.0),
    ((dms2float(20.5, seconds=21.2), dms2float(20.5, seconds=21.4), 5), {},
     np.arange(7382120, 7382141, 5), 360000.0),
    # test threshold factor
    ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5),
     {'threshold_factor': 60}, np.arange(12301, 12310), 600.0),
    ((dms2float(20.5, seconds=11.2), dms2float(20.5, seconds=53.3), 5),
     {'threshold_factor': 1}, np.arange(20502, 20517, 2), 1000.0),
])
def test_select_step360(args, kwargs, expected_levels, expected_factor):
    levels, n, factor = select_step360(*args, **kwargs)

    assert n == len(levels)
    np.testing.assert_array_equal(levels, expected_levels)
    assert factor == expected_factor


@pytest.mark.parametrize('direction, factor, values', [
    ("left", 60, [0, -30, -60]),
    ("left", 600, [12301, 12302, 12303]),
    ("left", 3600, [0, -30, -60]),
    ("left", 36000, [738210, 738215, 738220]),
    ("left", 360000, [7382120, 7382125, 7382130]),
    ("left", 1., [45, 46, 47]),
    ("left", 10., [452, 453, 454]),
])
def test_formatters(direction, factor, values):
    fmt = FormatterDMS()
    result = fmt(direction, factor, values)

    prev_degree = prev_minute = prev_second = None
    for tick, value in zip(result, values):
        m = DMS_RE.match(tick)
        assert m is not None, '"%s" is not an expected tick format.' % (tick, )

        sign = sum(m.group(sign + '_sign') is not None
                   for sign in ('degree', 'minute', 'second'))
        assert sign <= 1, \
            'Only one element of tick "%s" may have a sign.' % (tick, )
        sign = 1 if sign == 0 else -1

        degree = float(m.group('degree') or prev_degree or 0)
        minute = float(m.group('minute') or prev_minute or 0)
        second = float(m.group('second') or prev_second or 0)
        assert sign * dms2float(degree, minute, second) == value / factor, \
            '"%s" does not match expected tick value.' % (tick, )

        prev_degree = degree
        prev_minute = minute
        prev_second = second
