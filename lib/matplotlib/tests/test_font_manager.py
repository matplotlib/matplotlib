from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import sys
import tempfile
import warnings

import pytest

from matplotlib.font_manager import (
    findfont, FontProperties, fontManager, json_dump, json_load, get_font,
    get_fontconfig_fonts, is_opentype_cff_font, fontManager as fm)
from matplotlib import rc_context


def test_font_priority():
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        font = findfont(
            FontProperties(family=["sans-serif"]))
    assert os.path.basename(font) == 'cmmi10.ttf'

    # Smoketest get_charmap, which isn't used internally anymore
    font = get_font(font)
    cmap = font.get_charmap()
    assert len(cmap) == 131
    assert cmap[8729] == 30


def test_json_serialization():
    # on windows, we can't open a file twice, so save the name and unlink
    # manually...
    try:
        name = None
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            name = temp.name
        json_dump(fontManager, name)
        copy = json_load(name)
    finally:
        if name and os.path.exists(name):
            os.remove(name)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'findfont: Font family.*not found')
        for prop in ({'family': 'STIXGeneral'},
                     {'family': 'Bitstream Vera Sans', 'weight': 700},
                     {'family': 'no such font family'}):
            fp = FontProperties(**prop)
            assert (fontManager.findfont(fp, rebuild_if_missing=False) ==
                    copy.findfont(fp, rebuild_if_missing=False))


def test_otf():
    fname = '/usr/share/fonts/opentype/freefont/FreeMono.otf'
    if os.path.exists(fname):
        assert is_opentype_cff_font(fname)

    otf_files = [f for f in fm.ttffiles if 'otf' in f]
    for f in otf_files:
        with open(f, 'rb') as fd:
            res = fd.read(4) == b'OTTO'
        assert res == is_opentype_cff_font(f)


@pytest.mark.skipif(sys.platform == 'win32', reason='no fontconfig on Windows')
def test_get_fontconfig_fonts():
    assert len(get_fontconfig_fonts()) > 1
