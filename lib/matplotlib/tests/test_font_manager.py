from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
import six

import os
import sys

from matplotlib.font_manager import (
    findfont, FontProperties, get_font,
    get_fontconfig_fonts, is_opentype_cff_font, fontManager as fm)
from matplotlib import rc_context


def test_font_priority():
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        font = findfont(
            FontProperties(family=["sans-serif"]))
    assert_equal(os.path.basename(font), 'cmmi10.ttf')

    # Smoketest get_charmap, which isn't used internally anymore
    font = get_font(font)
    cmap = font.get_charmap()
    assert len(cmap) == 131
    assert cmap[8729] == 30


def test_otf():
    fname = '/usr/share/fonts/opentype/freefont/FreeMono.otf'
    if os.path.exists(fname):
        assert is_opentype_cff_font(fname)

    otf_files = [f for f in fm.ttffiles if 'otf' in f]
    for f in otf_files:
        with open(f, 'rb') as fd:
            res = fd.read(4) == b'OTTO'
        assert res == is_opentype_cff_font(f)


def test_get_fontconfig_fonts():
    assert sys.platform == 'win32' or len(get_fontconfig_fonts()) > 1
