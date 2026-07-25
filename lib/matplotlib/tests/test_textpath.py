import copy

import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath, TextToPath


def test_glyphs_load_their_own_outline():
    # Laying out no longer leaves a glyph in the font's slot, so each outline
    # must be loaded where it is read, not taken from whatever was there.
    font = fm.get_font(fm.findfont('DejaVu Sans'))
    _, glyph_map, _ = TextToPath().get_glyphs_with_font(font, 'lM')
    (l_verts, _), (m_verts, _) = glyph_map.values()
    assert len(l_verts) != len(m_verts)


def test_copy():
    tp = TextPath((0, 0), ".")
    assert copy.deepcopy(tp).vertices is not tp.vertices
    assert (copy.deepcopy(tp).vertices == tp.vertices).all()
    assert copy.copy(tp).vertices is tp.vertices
