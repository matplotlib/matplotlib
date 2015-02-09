from __future__ import print_function
from nose.tools import assert_raises
from matplotlib import ft2font
from matplotlib.testing.decorators import knownfailureif
import sys

def test_printf_buffer():
    """Tests Printf for buffer overrun."""
    # Use ft2font.FT2Font, which indirectly calls the Printf function in
    # mplutils.cpp.
    # Expect a RuntimeError, since the font is not found.
    assert_raises(RuntimeError, ft2font.FT2Font, 'x' * 2048)
