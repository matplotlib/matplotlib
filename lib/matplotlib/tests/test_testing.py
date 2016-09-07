from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path

from matplotlib.testing import closed_tempfile


def test_closed_tempfile():
    with closed_tempfile(".txt") as fname:
        assert os.path.exists(fname)
        assert fname.endswith(".txt")
        name = fname
    assert os.path.exists(name)


def test_closed_tempfile_text():
    text = "This is a test"
    with closed_tempfile(".txt", text=text) as f:
        with open(f, "rt") as g:
            assert text == g.read()
