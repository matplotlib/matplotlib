from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.afm as afm
import six


def test_nonascii_str():
    # This tests that we also decode bytes as utf-8 properly.
    # Else, font files with non ascii caracters fail to load.

    if six.PY3:
        string = "привет".encode("utf8")
    else:
        string = "привет"
    afm._to_str(string)
