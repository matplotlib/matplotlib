from contextlib import AbstractContextManager
from io import BytesIO
from typing import Any, NamedTuple

from matplotlib.backend_bases import GCT, RendererBase
from matplotlib.font_manager import FontPath
from matplotlib.ft2font import CharacterCodeType, FT2Font, GlyphIndexType

class SubsetResults(NamedTuple):
    font: Any  # fontTools.ttLib.TTFont
    glyph_index_map: dict[GlyphIndexType, GlyphIndexType]

def get_glyphs_subset(
    fontfile: FontPath, glyphs: set[GlyphIndexType]
) -> AbstractContextManager[SubsetResults]: ...
def font_as_file(font: Any) -> BytesIO: ...

class GlyphMap:
    def __init__(self) -> None: ...
    def get(
        self, charcodes: str, glyph_index: GlyphIndexType
    ) -> tuple[int, CharacterCodeType] | None: ...
    def iget(
        self, subset: int, subset_charcode: CharacterCodeType
    ) -> tuple[str, GlyphIndexType]: ...
    def add(
        self,
        charcode: str,
        glyph_index: GlyphIndexType,
        subset: int,
        subset_charcode: CharacterCodeType,
    ) -> None: ...

class CharacterTracker:
    used: dict[str, list[dict[CharacterCodeType, GlyphIndexType]]]
    glyph_maps: dict[str, GlyphMap]
    subset_size: int
    def __init__(self, subset_size: int = ...) -> None: ...
    def track(
        self,
        font: FT2Font,
        s: str,
        features: tuple[str, ...] | None = ...,
        language: str | tuple[tuple[str, int, int], ...] | None = ...,
    ) -> list[tuple[int, CharacterCodeType]]: ...
    def track_glyph(
        self,
        font: FT2Font,
        chars: str | CharacterCodeType,
        glyph: GlyphIndexType,
    ) -> tuple[int, CharacterCodeType]: ...
    def subset_to_unicode(
        self, fontname: str, subset: int, subset_charcode: CharacterCodeType
    ) -> str: ...

class RendererPDFPSBase(RendererBase[GCT]):
    width: float
    height: float
    def __init__(self, width: float, height: float) -> None: ...
