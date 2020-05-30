import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager

import pytest


def test_fontconfig_preamble():
    """
    Test that the preamble is included in _fontconfig
    """
    plt.rcParams['text.usetex'] = True

    tm1 = TexManager()
    font_config1 = tm1.get_font_config()

    plt.rcParams['text.latex.preamble'] = '\\usepackage{txfonts}'
    tm2 = TexManager()
    font_config2 = tm2.get_font_config()

    assert font_config1 != font_config2


@pytest.mark.parametrize('textcomp_full, expected_result',
                         [(True, 'usepackage[full]{textcomp}'),
                          (False, 'usepackage{textcomp}')])
def test_textcomp_full(textcomp_full, expected_result):
    """
    Addresses issue #9118.
    See https://github.com/matplotlib/matplotlib/issues/9118.

    Test that the [full] option is set for the textcomp package
    in _font_preamble if the rcParam text.latex.textcomp_full
    is set to True, and that the option is not set otherwise.
    """
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.textcomp_full'] = textcomp_full

    tm = TexManager()
    tm.get_font_config()  # This also sets tm1._font_preamble

    font_preamble = tm.get_font_preamble()
    if 'textcomp' not in font_preamble:
        # Do not test for anything if the textcomp package
        # is not in the font preamble.
        return

    assert expected_result in font_preamble
