"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""
from collections.abc import Hashable, Sequence
import pathlib
from typing import Any, Literal, TypeVar

from . import path
from ._enums import JoinStyle, CapStyle
from .markers import MarkerStyle

# The following are type aliases. Once python 3.9 is dropped, they should be annotated
# using ``typing.TypeAlias``.

RGBColorType = tuple[float, float, float] | str
RGBAColorType = (
    str |  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float] |
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float] |
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float]
)

ColorType = RGBColorType | RGBAColorType

RGBColourType = RGBColorType
RGBAColourType = RGBAColorType
ColourType = ColorType

LineStyleType = str | tuple[float, Sequence[float]]
DrawStyleType = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
MarkEveryType = (
    None |
    int | tuple[int, int] | slice | list[int] |
    float | tuple[float, float] |
    list[bool]
)

MarkerType = str | path.Path | MarkerStyle
FillStyleType = Literal["full", "left", "right", "bottom", "top", "none"]
JoinStyleType = JoinStyle | Literal["miter", "round", "bevel"]
CapStyleType = CapStyle | Literal["butt", "projecting", "round"]

RcStyleType = (
    str |
    dict[str, Any] |
    pathlib.Path |
    Sequence[str | pathlib.Path | dict[str, Any]]
)

_HT = TypeVar("_HT", bound=Hashable)
HashableList = list[_HT | "HashableList[_HT]"]
"""A nested list of Hashable values."""
