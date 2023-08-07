import os
from matplotlib.font_manager import FontProperties

# Re-exported API from _mathtext.
from ._mathtext import (
    RasterParse as RasterParse,
    VectorParse as VectorParse,
    get_unicode_index as get_unicode_index,
)

from typing import IO, Literal
from matplotlib.typing import ColorType

class MathTextParser:
    def __init__(self, output: Literal["path", "agg", "raster", "macosx"]) -> None: ...
    def parse(
        self, s: str, dpi: float = ..., prop: FontProperties | None = ..., *, antialiased: bool | None = ...
    ) -> RasterParse | VectorParse: ...

def math_to_image(
    s: str,
    filename_or_obj: str | os.PathLike | IO,
    prop: FontProperties | None = ...,
    dpi: float | None = ...,
    format: str | None = ...,
    *,
    color: ColorType | None = ...
) -> float: ...
