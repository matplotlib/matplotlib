import io
import os
from ._mathtext import RasterParse, VectorParse, get_unicode_index
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Image, LOAD_NO_HINTING

from typing import Literal
from matplotlib.typing import ColorType

class MathTextParser:
    def __init__(self, output: Literal["path", "raster", "macosx"]) -> None: ...
    def parse(self, s: str, dpi: float = ..., prop: FontProperties | None = ...): ...

def math_to_image(
    s: str,
    filename_or_obj: str | os.PathLike | io.FileIO,
    prop: FontProperties | None = ...,
    dpi: float | None = ...,
    format: str | None = ...,
    *,
    color: ColorType | None = ...
): ...
