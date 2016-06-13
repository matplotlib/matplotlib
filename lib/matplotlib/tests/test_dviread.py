from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from nose.tools import assert_equal
import matplotlib.dviread as dr
import os.path

original_find_tex_file = dr.find_tex_file

def setup():
    dr.find_tex_file = lambda x: x

def teardown():
    dr.find_tex_file = original_find_tex_file

def test_PsfontsMap():
    filename = os.path.join(
        os.path.dirname(__file__),
        'baseline_images', 'dviread', 'test.map')
    fontmap = dr.PsfontsMap(filename)
    # Check all properties of a few fonts
    for n in [1, 2, 3, 4, 5]:
        key = 'TeXfont%d' % n
        entry = fontmap[key]
        assert_equal(entry.texname, key)
        assert_equal(entry.psname, 'PSfont%d' % n)
        if n not in [3, 5]:
            assert_equal(entry.encoding, 'font%d.enc' % n)
        elif n == 3:
            assert_equal(entry.encoding, 'enc3.foo')
        # We don't care about the encoding of TeXfont5, which specifies
        # multiple encodings.
        if n not in [1, 5]:
            assert_equal(entry.filename, 'font%d.pfa' % n)
        else:
            assert_equal(entry.filename, 'font%d.pfb' % n)
        if n == 4:
            assert_equal(entry.effects, {'slant': -0.1, 'extend': 2.2})
        else:
            assert_equal(entry.effects, {})
    # Some special cases
    entry = fontmap['TeXfont6']
    assert_equal(entry.filename, None)
    assert_equal(entry.encoding, None)
    entry = fontmap['TeXfont7']
    assert_equal(entry.filename, None)
    assert_equal(entry.encoding, 'font7.enc')
    entry = fontmap['TeXfont8']
    assert_equal(entry.filename, 'font8.pfb')
    assert_equal(entry.encoding, None)
    entry = fontmap['TeXfont9']
    assert_equal(entry.filename, '/absolute/font9.pfb')
