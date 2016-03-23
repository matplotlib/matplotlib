from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
from matplotlib.externals import six

import os
import tempfile
import warnings

from matplotlib.font_manager import (
    findfont, FontProperties, fontManager, json_dump, json_load, get_font)
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
            assert_equal(fontManager.findfont(fp, rebuild_if_missing=False),
                         copy.findfont(fp, rebuild_if_missing=False))
