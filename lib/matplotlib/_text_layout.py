"""
Text layouting utilities.
"""

from .ft2font import KERNING_DEFAULT, LOAD_NO_HINTING


def layout(string, font, *, kern_mode=KERNING_DEFAULT):
    """
    Render *string* with *font*.  For each character in *string*, yield a
    (glyph-index, x-position) pair.  When such a pair is yielded, the font's
    glyph is set to the corresponding character.

    Parameters
    ----------
    string : str
        The string to be rendered.
    font : FT2Font
        The font.
    kern_mode : int
        A FreeType kerning mode.

    Yields
    ------
    glyph_index : int
    x_position : float
    """
    x = 0
    last_glyph_idx = None
    for char in string:
        glyph_idx = font.get_char_index(ord(char))
        kern = (font.get_kerning(last_glyph_idx, glyph_idx, kern_mode)
                if last_glyph_idx is not None else 0) / 64
        x += kern
        glyph = font.load_glyph(glyph_idx, flags=LOAD_NO_HINTING)
        yield glyph_idx, x
        x += glyph.linearHoriAdvance / 65536
        last_glyph_idx = glyph_idx
