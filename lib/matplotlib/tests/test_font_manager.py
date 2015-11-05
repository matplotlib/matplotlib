from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
from matplotlib.externals import six

import os

from matplotlib.font_manager import (findfont, FontProperties, get_font)
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
