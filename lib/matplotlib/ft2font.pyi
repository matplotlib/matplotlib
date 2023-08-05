from typing import BinaryIO, Literal

import numpy as np
from numpy.typing import NDArray

__freetype_build_type__: str
__freetype_version__: str
BOLD: int
EXTERNAL_STREAM: int
FAST_GLYPHS: int
FIXED_SIZES: int
FIXED_WIDTH: int
GLYPH_NAMES: int
HORIZONTAL: int
ITALIC: int
KERNING: int
KERNING_DEFAULT: int
KERNING_UNFITTED: int
KERNING_UNSCALED: int
LOAD_CROP_BITMAP: int
LOAD_DEFAULT: int
LOAD_FORCE_AUTOHINT: int
LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH: int
LOAD_IGNORE_TRANSFORM: int
LOAD_LINEAR_DESIGN: int
LOAD_MONOCHROME: int
LOAD_NO_AUTOHINT: int
LOAD_NO_BITMAP: int
LOAD_NO_HINTING: int
LOAD_NO_RECURSE: int
LOAD_NO_SCALE: int
LOAD_PEDANTIC: int
LOAD_RENDER: int
LOAD_TARGET_LCD: int
LOAD_TARGET_LCD_V: int
LOAD_TARGET_LIGHT: int
LOAD_TARGET_MONO: int
LOAD_TARGET_NORMAL: int
LOAD_VERTICAL_LAYOUT: int
MULTIPLE_MASTERS: int
SCALABLE: int
SFNT: int
VERTICAL: int

class FT2Font:
    ascender: int
    bbox: tuple[int, int, int, int]
    descender: int
    face_flags: int
    family_name: str
    fname: str
    height: int
    max_advance_height: int
    max_advance_width: int
    num_charmaps: int
    num_faces: int
    num_fixed_sizes: int
    num_glyphs: int
    postscript_name: str
    scalable: bool
    style_flags: int
    style_name: str
    underline_position: int
    underline_thickness: int
    units_per_EM: int

    def __init__(
        self,
        filename: str | BinaryIO,
        hinting_factor: int = ...,
        *,
        _fallback_list: list[FT2Font] | None = ...,
        _kerning_factor: int = ...
    ) -> None: ...
    def _get_fontmap(self, string: str) -> dict[str, FT2Font]: ...
    def clear(self) -> None: ...
    def draw_glyph_to_bitmap(
        self, image: FT2Image, x: float, y: float, glyph: Glyph, antialiased: bool = ...
    ) -> None: ...
    def draw_glyphs_to_bitmap(self, antialiased: bool = ...) -> None: ...
    def get_bitmap_offset(self) -> tuple[int, int]: ...
    def get_char_index(self, codepoint: int) -> int: ...
    def get_charmap(self) -> dict[int, int]: ...
    def get_descent(self) -> int: ...
    def get_glyph_name(self, index: int) -> str: ...
    def get_image(self) -> NDArray[np.uint8]: ...
    def get_kerning(self, left: int, right: int, mode: int) -> int: ...
    def get_name_index(self, name: str) -> int: ...
    def get_num_glyphs(self) -> int: ...
    def get_path(self) -> tuple[NDArray[np.float64], NDArray[np.int8]]: ...
    def get_ps_font_info(
        self,
    ) -> tuple[str, str, str, str, str, int, int, int, int]: ...
    def get_sfnt(self) -> dict[tuple[int, int, int, int], bytes]: ...
    def get_sfnt_table(
        self, name: Literal["head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt"]
    ) -> dict[str, tuple[int, int, int, int] | tuple[int, int] | int | bytes]: ...
    def get_width_height(self) -> tuple[int, int]: ...
    def get_xys(self, antialiased: bool = ...) -> NDArray[np.float64]: ...
    def load_char(self, charcode: int, flags: int = ...) -> Glyph: ...
    def load_glyph(self, glyphindex: int, flags: int = ...) -> Glyph: ...
    def select_charmap(self, i: int) -> None: ...
    def set_charmap(self, i: int) -> None: ...
    def set_size(self, ptsize: float, dpi: float) -> None: ...
    def set_text(
        self, string: str, angle: float = ..., flags: int = ...
    ) -> NDArray[np.float64]: ...

class FT2Image:  # TODO: When updating mypy>=1.4, subclass from Buffer.
    def __init__(self, width: float, height: float) -> None: ...
    def draw_rect(self, x0: float, y0: float, x1: float, y1: float) -> None: ...
    def draw_rect_filled(self, x0: float, y0: float, x1: float, y1: float) -> None: ...

class Glyph:
    width: int
    height: int
    horiBearingX: int
    horiBearingY: int
    horiAdvance: int
    linearHoriAdvance: int
    vertBearingX: int
    vertBearingY: int
    vertAdvance: int

    def bbox(self) -> tuple[int, int, int, int]: ...
