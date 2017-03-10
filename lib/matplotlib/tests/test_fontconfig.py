from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
from matplotlib.externals import six

import os

from matplotlib.font_manager import FontProperties
from matplotlib.fontconfig_pattern import generate_fontconfig_pattern
from matplotlib.fontconfig_pattern import parse_fontconfig_pattern


def check_fonts_same(f1, f2):
    assert_equal(f1.get_family(), f2.get_family())
    assert_equal(f1.get_style(), f2.get_style())
    assert_equal(f1.get_variant(), f2.get_variant())
    assert_equal(f1.get_weight(), f2.get_weight())
    assert_equal(f1.get_stretch(), f2.get_stretch())
    assert_equal(f1.get_size(), f2.get_size())


def do_test(font, expected):
    # First convert the mpl FontProperties to a fontconfig string
    fc = generate_fontconfig_pattern(font)

    # make sure it is what we expected
    assert_equal(fc, expected)

    # now make sure we can convert it back
    newfont = parse_fontconfig_pattern(fc)

    # ensure the new font property maches the old one
    check_fonts_same(font, newfont)


def test_fontconfig():
    s1 ="u'sans\\-serif:style=normal:variant=normal:weight=normal:stretch=normal:size=12.0'"
    f1 = FontProperties()
    do_test(f1, s1)

    s2 ="u'serif:style=normal:variant=normal:weight=normal:stretch=normal:size=12.0'"
    f2 = FontProperties(family = 'serif')
    do_test(f2, s2)

    s1 ="u'sans\\-serif:style=normal:variant=normal:weight=bold:stretch=normal:size=24.0'"
    f3 = FontProperties(size=24, weight="bold")
    f3.set_family(None)
    do_test(f3, s3)

