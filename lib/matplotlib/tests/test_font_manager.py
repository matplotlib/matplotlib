from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pytest

from matplotlib.font_manager import (
    findfont, FontProperties, fontManager, json_dump, json_load, get_font,
    get_fontconfig_fonts, is_opentype_cff_font)
from matplotlib import rc_context

has_fclist = shutil.which('fc-list') is not None


def test_font_priority():
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        font = findfont(FontProperties(family=["sans-serif"]))
    assert Path(font).name == 'cmmi10.ttf'

    # Smoketest get_charmap, which isn't used internally anymore
    font = get_font(font)
    cmap = font.get_charmap()
    assert len(cmap) == 131
    assert cmap[8729] == 30


def test_score_weight():
    assert 0 == fontManager.score_weight("regular", "regular")
    assert 0 == fontManager.score_weight("bold", "bold")
    assert (0 < fontManager.score_weight(400, 400) <
            fontManager.score_weight("normal", "bold"))
    assert (0 < fontManager.score_weight("normal", "regular") <
            fontManager.score_weight("normal", "bold"))
    assert (fontManager.score_weight("normal", "regular") ==
            fontManager.score_weight(400, 400))


def test_json_serialization(tmpdir):
    # Can't open a NamedTemporaryFile twice on Windows, so use a temporary
    # directory instead.
    path = Path(tmpdir, "fontlist.json")
    json_dump(fontManager, path)
    copy = json_load(path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'findfont: Font family.*not found')
        for prop in ({'family': 'STIXGeneral'},
                     {'family': 'Bitstream Vera Sans', 'weight': 700},
                     {'family': 'no such font family'}):
            fp = FontProperties(**prop)
            assert (fontManager.findfont(fp, rebuild_if_missing=False) ==
                    copy.findfont(fp, rebuild_if_missing=False))


def test_otf():
    fname = '/usr/share/fonts/opentype/freefont/FreeMono.otf'
    if Path(fname).exists():
        assert is_opentype_cff_font(fname)
    for f in fontManager.ttflist:
        if 'otf' in f.fname:
            with open(f.fname, 'rb') as fd:
                res = fd.read(4) == b'OTTO'
            assert res == is_opentype_cff_font(f.fname)


@pytest.mark.skipif(not has_fclist, reason='no fontconfig installed')
def test_get_fontconfig_fonts():
    assert len(get_fontconfig_fonts()) > 1


@pytest.mark.parametrize('factor', [2, 4, 6, 8])
def test_hinting_factor(factor):
    font = findfont(FontProperties(family=["sans-serif"]))

    font1 = get_font(font, hinting_factor=1)
    font1.clear()
    font1.set_size(12, 100)
    font1.set_text('abc')
    expected = font1.get_width_height()

    hinted_font = get_font(font, hinting_factor=factor)
    hinted_font.clear()
    hinted_font.set_size(12, 100)
    hinted_font.set_text('abc')
    # Check that hinting only changes text layout by a small (10%) amount.
    np.testing.assert_allclose(hinted_font.get_width_height(), expected,
                               rtol=0.1)


@pytest.mark.skipif(sys.platform != "win32",
                   reason="Need Windows font to test against")
def test_utf16m_sfnt():
    segoe_ui_semibold = None
    for f in fontManager.ttflist:
        # seguisbi = Microsoft Segoe UI Semibold
        if f.fname[-12:] == "seguisbi.ttf":
            segoe_ui_semibold = f
            break
    else:
        pytest.xfail(reason="Couldn't find font to test against.")

    # Check that we successfully read the "semibold" from the font's
    # sfnt table and set its weight accordingly
    assert segoe_ui_semibold.weight == "semibold"
