from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from nose.tools import assert_equal
import six

import os

from matplotlib.font_manager import findfont, FontProperties
from matplotlib import rc_context


def test_font_priority():
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        font = findfont(
            FontProperties(family=["sans-serif"]))
    assert_equal(os.path.basename(font), 'cmmi10.ttf')


def test_font_ttc():
    # the font should be available(ttf-wqy-zenhei in ubuntu)
    font = findfont(
        FontProperties(family=["WenQuanYi Zen Hei"]))
    assert_equal(os.path.basename(font), 'wqy-zenhei.ttc')

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
