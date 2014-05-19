from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
import six

import os

from matplotlib.font_manager import findfont, FontProperties, _rebuild
from matplotlib import rc_context


def test_font_priority():
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        # force the font manager to rebuild it self
        _rebuild()
        font = findfont(
            FontProperties(family=["sans-serif"]))
    assert_equal(os.path.basename(font), 'cmmi10.ttf')
    # force it again
    _rebuild()
