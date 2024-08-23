import itertools
import io
from pathlib import Path

import numpy as np
import pytest

from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def test_ft2image_draw_rect_filled():
    width = 23
    height = 42
    for x0, y0, x1, y1 in itertools.product([1, 100], [2, 200], [4, 400], [8, 800]):
        im = ft2font.FT2Image(width, height)
        im.draw_rect_filled(x0, y0, x1, y1)
        a = np.asarray(im)
        assert a.dtype == np.uint8
        assert a.shape == (height, width)
        if x0 == 100 or y0 == 200:
            # All the out-of-bounds starts should get automatically clipped.
            assert np.sum(a) == 0
        else:
            # Otherwise, ends are clipped to the dimension, but are also _inclusive_.
            filled = (min(x1 + 1, width) - x0) * (min(y1 + 1, height) - y0)
            assert np.sum(a) == 255 * filled


def test_ft2font_dejavu_attrs():
    file = fm.findfont('DejaVu Sans')
    font = ft2font.FT2Font(file)
    assert font.fname == file
    # Names extracted from FontForge: Font Information → PS Names tab.
    assert font.postscript_name == 'DejaVuSans'
    assert font.family_name == 'DejaVu Sans'
    assert font.style_name == 'Book'
    assert font.num_faces == 1  # Single TTF.
    assert font.num_glyphs == 6241  # From compact encoding view in FontForge.
    assert font.num_fixed_sizes == 0  # All glyphs are scalable.
    assert font.num_charmaps == 5
    # Other internal flags are set, so only check the ones we're allowed to test.
    expected_flags = (ft2font.SCALABLE | ft2font.SFNT | ft2font.HORIZONTAL |
                      ft2font.KERNING | ft2font.GLYPH_NAMES)
    assert (font.face_flags & expected_flags) == expected_flags
    assert font.style_flags == 0  # Not italic or bold.
    assert font.scalable
    # From FontForge: Font Information → General tab → entry name below.
    assert font.units_per_EM == 2048  # Em Size.
    assert font.underline_position == -175  # Underline position.
    assert font.underline_thickness == 90  # Underline height.
    # From FontForge: Font Information → OS/2 tab → Metrics tab → entry name below.
    assert font.ascender == 1901  # HHead Ascent.
    assert font.descender == -483  # HHead Descent.
    # Unconfirmed values.
    assert font.height == 2384
    assert font.max_advance_width == 3838
    assert font.max_advance_height == 2384
    assert font.bbox == (-2090, -948, 3673, 2524)


def test_ft2font_cm_attrs():
    file = fm.findfont('cmtt10')
    font = ft2font.FT2Font(file)
    assert font.fname == file
    # Names extracted from FontForge: Font Information → PS Names tab.
    assert font.postscript_name == 'Cmtt10'
    assert font.family_name == 'cmtt10'
    assert font.style_name == 'Regular'
    assert font.num_faces == 1  # Single TTF.
    assert font.num_glyphs == 133  # From compact encoding view in FontForge.
    assert font.num_fixed_sizes == 0  # All glyphs are scalable.
    assert font.num_charmaps == 2
    # Other internal flags are set, so only check the ones we're allowed to test.
    expected_flags = (ft2font.SCALABLE | ft2font.SFNT | ft2font.HORIZONTAL |
                      ft2font.GLYPH_NAMES)
    assert (font.face_flags & expected_flags) == expected_flags, font.face_flags
    assert font.style_flags == 0  # Not italic or bold.
    assert font.scalable
    # From FontForge: Font Information → General tab → entry name below.
    assert font.units_per_EM == 2048  # Em Size.
    assert font.underline_position == -143  # Underline position.
    assert font.underline_thickness == 20  # Underline height.
    # From FontForge: Font Information → OS/2 tab → Metrics tab → entry name below.
    assert font.ascender == 1276  # HHead Ascent.
    assert font.descender == -489  # HHead Descent.
    # Unconfirmed values.
    assert font.height == 1765
    assert font.max_advance_width == 1536
    assert font.max_advance_height == 1765
    assert font.bbox == (-12, -477, 1280, 1430)


def test_ft2font_stix_bold_attrs():
    file = fm.findfont('STIXSizeTwoSym:bold')
    font = ft2font.FT2Font(file)
    assert font.fname == file
    # Names extracted from FontForge: Font Information → PS Names tab.
    assert font.postscript_name == 'STIXSizeTwoSym-Bold'
    assert font.family_name == 'STIXSizeTwoSym'
    assert font.style_name == 'Bold'
    assert font.num_faces == 1  # Single TTF.
    assert font.num_glyphs == 20  # From compact encoding view in FontForge.
    assert font.num_fixed_sizes == 0  # All glyphs are scalable.
    assert font.num_charmaps == 3
    # Other internal flags are set, so only check the ones we're allowed to test.
    expected_flags = (ft2font.SCALABLE | ft2font.SFNT | ft2font.HORIZONTAL |
                      ft2font.GLYPH_NAMES)
    assert (font.face_flags & expected_flags) == expected_flags, font.face_flags
    assert font.style_flags == ft2font.BOLD
    assert font.scalable
    # From FontForge: Font Information → General tab → entry name below.
    assert font.units_per_EM == 1000  # Em Size.
    assert font.underline_position == -133  # Underline position.
    assert font.underline_thickness == 20  # Underline height.
    # From FontForge: Font Information → OS/2 tab → Metrics tab → entry name below.
    assert font.ascender == 2095  # HHead Ascent.
    assert font.descender == -404  # HHead Descent.
    # Unconfirmed values.
    assert font.height == 2499
    assert font.max_advance_width == 1130
    assert font.max_advance_height == 2499
    assert font.bbox == (4, -355, 1185, 2095)


def test_ft2font_invalid_args(tmp_path):
    # filename argument.
    with pytest.raises(TypeError, match='to a font file or a binary-mode file object'):
        ft2font.FT2Font(None)
    file = tmp_path / 'invalid-font.ttf'
    file.write_text('This is not a valid font file.')
    with (pytest.raises(TypeError, match='to a font file or a binary-mode file object'),
          file.open('rt') as fd):
        ft2font.FT2Font(fd)
    with (pytest.raises(TypeError, match='to a font file or a binary-mode file object'),
          file.open('wt') as fd):
        ft2font.FT2Font(fd)
    with (pytest.raises(TypeError, match='to a font file or a binary-mode file object'),
          file.open('wb') as fd):
        ft2font.FT2Font(fd)

    file = fm.findfont('DejaVu Sans')

    # hinting_factor argument.
    with pytest.raises(TypeError, match='cannot be interpreted as an integer'):
        ft2font.FT2Font(file, 1.3)
    with pytest.raises(ValueError, match='hinting_factor must be greater than 0'):
        ft2font.FT2Font(file, 0)

    with pytest.raises(TypeError, match='Fallback list must be a list'):
        # failing to be a list will fail before the 0
        ft2font.FT2Font(file, _fallback_list=(0,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match='Fallback fonts must be FT2Font objects.'):
        ft2font.FT2Font(file, _fallback_list=[0])  # type: ignore[list-item]

    # kerning_factor argument.
    with pytest.raises(TypeError, match='cannot be interpreted as an integer'):
        ft2font.FT2Font(file, _kerning_factor=1.3)


@pytest.mark.parametrize('family_name, file_name',
                          [("WenQuanYi Zen Hei",  "wqy-zenhei.ttc"),
                           ("Noto Sans CJK JP", "NotoSansCJK.ttc"),
                           ("Noto Sans TC", "NotoSansTC-Regular.otf")]
                         )
def test_fallback_smoke(family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    if Path(fm.findfont(fp)).name != file_name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")
    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(4.75, 1.85))
    fig.text(0.05, 0.45, "There are 几个汉字 in between!",
             family=['DejaVu Sans', family_name])
    fig.text(0.05, 0.85, "There are 几个汉字 in between!",
             family=[family_name])

    # TODO enable fallback for other backends!
    for fmt in ['png', 'raw']:  # ["svg", "pdf", "ps"]:
        fig.savefig(io.BytesIO(), format=fmt)


@pytest.mark.parametrize('family_name, file_name',
                         [("WenQuanYi Zen Hei",  "wqy-zenhei"),
                          ("Noto Sans CJK JP", "NotoSansCJK"),
                          ("Noto Sans TC", "NotoSansTC-Regular.otf")]
                         )
@check_figures_equal(extensions=["png", "pdf", "eps", "svg"])
def test_font_fallback_chinese(fig_test, fig_ref, family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    if file_name not in Path(fm.findfont(fp)).name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    text = ["There are", "几个汉字", "in between!"]

    plt.rcParams["font.size"] = 20
    test_fonts = [["DejaVu Sans", family_name]] * 3
    ref_fonts = [["DejaVu Sans"], [family_name], ["DejaVu Sans"]]

    for j, (txt, test_font, ref_font) in enumerate(
            zip(text, test_fonts, ref_fonts)
    ):
        fig_ref.text(0.05, .85 - 0.15*j, txt, family=ref_font)
        fig_test.text(0.05, .85 - 0.15*j, txt, family=test_font)


@pytest.mark.parametrize("font_list",
                          [['DejaVu Serif', 'DejaVu Sans'],
                           ['DejaVu Sans Mono']],
                         ids=["two fonts", "one font"])
def test_fallback_missing(recwarn, font_list):
    fig = plt.figure()
    fig.text(.5, .5, "Hello 🙃 World!", family=font_list)
    fig.canvas.draw()
    assert all(isinstance(warn.message, UserWarning) for warn in recwarn)
    # not sure order is guaranteed on the font listing so
    assert recwarn[0].message.args[0].startswith(
           "Glyph 128579 (\\N{UPSIDE-DOWN FACE}) missing from font(s)")
    assert all([font in recwarn[0].message.args[0] for font in font_list])


@pytest.mark.parametrize(
    "family_name, file_name",
    [
        ("WenQuanYi Zen Hei", "wqy-zenhei"),
        ("Noto Sans CJK JP", "NotoSansCJK"),
        ("Noto Sans TC", "NotoSansTC-Regular.otf")
    ],
)
def test__get_fontmap(family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    found_file_name = Path(fm.findfont(fp)).name
    if file_name not in found_file_name:
        pytest.skip(f"Font {family_name} ({file_name}) is missing")

    text = "There are 几个汉字 in between!"
    ft = fm.get_font(
        fm.fontManager._find_fonts_by_props(
            fm.FontProperties(family=["DejaVu Sans", family_name])
        )
    )
    fontmap = ft._get_fontmap(text)
    for char, font in fontmap.items():
        if ord(char) > 127:
            assert Path(font.fname).name == found_file_name
        else:
            assert Path(font.fname).name == "DejaVuSans.ttf"
