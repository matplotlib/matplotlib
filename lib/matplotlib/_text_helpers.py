"""
Low-level text helper utilities.
"""

from __future__ import annotations

from collections.abc import Iterator

from . import _api
from .ft2font import FT2Font, LayoutItem, LoadFlags


def warn_on_missing_glyph(codepoint: int, fontnames: str):
    _api.warn_external(
        f"Glyph {codepoint} "
        f"({chr(codepoint).encode('ascii', 'namereplace').decode('ascii')}) "
        f"missing from font(s) {fontnames}.")


def layout(string: str, font: FT2Font) -> Iterator[LayoutItem]:
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

    Yields
    ------
    LayoutItem
    """
    for raqm_item in font._layout(string, LoadFlags.NO_HINTING):
        raqm_item.ft_object.load_glyph(raqm_item.glyph_idx, flags=LoadFlags.NO_HINTING)
        yield raqm_item
