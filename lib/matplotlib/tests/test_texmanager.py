import pytest

from matplotlib import pyplot as plt, checkdep_usetex
from matplotlib.texmanager import TexManager


needs_usetex = pytest.mark.skipif(
    not checkdep_usetex(True),
    reason="This test needs a TeX installation")


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


@needs_usetex
def test_usetex_missing_underscore():
    """
    Test that failed TeX rendering due to an unescaped underscore has a
    custom error message.
    """
    with pytest.raises(RuntimeError,
                       match='caused by an unescaped underscore'):
        plt.text(0, 0, 'foo_bar', usetex=True)
        plt.draw()  # TeX rendering is done at draw time
