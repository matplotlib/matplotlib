# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.afm as afm


def test_nonascii_str():
    # This tests that we also decode bytes as utf-8 properly.
    # Else, font files with non ascii characters fail to load.
    inp_str = "привет"
    byte_str = inp_str.encode("utf8")

    ret = afm._to_str(byte_str)
    assert ret == inp_str
