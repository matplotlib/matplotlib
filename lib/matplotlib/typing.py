"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""
from collections.abc import Sequence
import pathlib
from typing import Any, Hashable, Literal, Union

from . import path
from .markers import MarkerStyle

# The following are type aliases. Once python 3.9 is dropped, they should be annotated
# using ``typing.TypeAlias`` and Unions should be converted to using ``|`` syntax.

ColorType = Union[tuple[float, float, float], tuple[float, float, float, float], str]
ColourType = ColorType

LineStyleType = Union[str, tuple[float, Sequence[float]]]
DrawStyleType = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
MarkEveryType = Union[
    None,
    int,
    tuple[int, int],
    slice,
    list[int],
    float,
    tuple[float, float],
    list[bool]
]

MarkerType = Union[str, path.Path, MarkerStyle]
FillStyleType = Literal["full", "left", "right", "bottom", "top", "none"]

RcStyleType = Union[
    str, dict[str, Any], pathlib.Path, list[Union[str, pathlib.Path, dict[str, Any]]]
]

HashableList = list[Union[Hashable, "HashableList"]]
