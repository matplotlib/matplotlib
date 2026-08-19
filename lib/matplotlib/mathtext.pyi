import os
from typing import IO, Literal, overload

from matplotlib.font_manager import FontProperties
from matplotlib.typing import ColorType

# Re-exported API from _mathtext.
from ._mathtext import (
    RasterParse as RasterParse,
    VectorParse as VectorParse,
    get_unicode_index as get_unicode_index,
)

class MathTextParser[ParseType: (RasterParse, VectorParse)]:
    @overload
    def __init__(self: MathTextParser[VectorParse], output: Literal["path"]) -> None: ...
    @overload
    def __init__(self: MathTextParser[RasterParse], output: Literal["agg", "raster", "macosx"]) -> None: ...
    def parse(
        self,
        s: str,
        dpi: float = ...,
        prop: FontProperties | None = ...,
        *,
        antialiased: bool | None = ...,
    ) -> ParseType: ...

def math_to_image(
    s: str,
    filename_or_obj: str | os.PathLike | IO,
    prop: FontProperties | None = ...,
    dpi: float | None = ...,
    format: str | None = ...,
    *,
    color: ColorType | None = ...
) -> float: ...
