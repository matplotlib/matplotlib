"""
Low-level text helper utilities.
"""

from __future__ import annotations

from . import _api
from .ft2font import Kerning, LoadFlags


def warn_on_missing_glyph(codepoint, fontnames):
    _api.warn_external(
        f"Glyph {codepoint} "
        f"({chr(codepoint).encode('ascii', 'namereplace').decode('ascii')}) "
        f"missing from font(s) {fontnames}.")


def layout(string, font, *, kern_mode=Kerning.DEFAULT):
    """
    Render *string* with *font*.

    For each character in *string*, yield a LayoutItem instance. When such an instance
    is yielded, the font's glyph is set to the corresponding character.

    Parameters
    ----------
    string : str
        The string to be rendered.
    font : FT2Font
        The font.
    kern_mode : Kerning
        A FreeType kerning mode.

    Yields
    ------
    LayoutItem
    """
    for raqm_item in font._layout(string, LoadFlags.NO_HINTING):
        raqm_item.ft_object.load_glyph(raqm_item.glyph_idx, flags=LoadFlags.NO_HINTING)
        yield raqm_item
