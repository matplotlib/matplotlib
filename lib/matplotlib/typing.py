"""
Typing support for Matplotlib

This module contains Type aliases which are useful for Matplotlib and potentially
downstream libraries.

.. warning::
    **Provisional status of typing**

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.
"""
from collections.abc import Hashable, Sequence
import pathlib
from typing import Any, Literal, TypeAlias, TypeVar, Union
from collections.abc import Callable

from . import path
from ._enums import JoinStyle, CapStyle
from .artist import Artist
from .backend_bases import RendererBase
from .markers import MarkerStyle
from .transforms import Bbox, Transform

RGBColorType: TypeAlias = tuple[float, float, float] | str
"""Any RGB color specification accepted by Matplotlib."""

RGBAColorType: TypeAlias = (
    str |  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    tuple[float, float, float, float] |
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    tuple[RGBColorType, float] |
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
    tuple[tuple[float, float, float, float], float]
)
"""Any RGBA color specification accepted by Matplotlib."""

ColorType: TypeAlias = RGBColorType | RGBAColorType
"""Any color specification accepted by Matplotlib. See :mpltype:`color`."""

RGBColourType: TypeAlias = RGBColorType
"""Alias of `.RGBColorType`."""

RGBAColourType: TypeAlias = RGBAColorType
"""Alias of `.RGBAColorType`."""

ColourType: TypeAlias = ColorType
"""Alias of `.ColorType`."""

LineStyleType: TypeAlias = (
    Literal["-", "solid", "--", "dashed", "-.", "dashdot", ":", "dotted",
            "", "none", " ", "None"] |
    tuple[float, Sequence[float]]
)
"""
Any line style specification accepted by Matplotlib.
See :doc:`/gallery/lines_bars_and_markers/linestyles`.
"""

DrawStyleType: TypeAlias = Literal["default", "steps", "steps-pre", "steps-mid",
                                   "steps-post"]
"""See :doc:`/gallery/lines_bars_and_markers/step_demo`."""

MarkEveryType: TypeAlias = (
    None |
    int | tuple[int, int] | slice | list[int] |
    float | tuple[float, float] |
    list[bool]
)
"""See :doc:`/gallery/lines_bars_and_markers/markevery_demo`."""

MarkerType: TypeAlias = (
    path.Path | MarkerStyle | str |  # str required for "$...$" marker
    Literal[
        ".", ",", "o", "v", "^", "<", ">",
        "1", "2", "3", "4", "8", "s", "p",
        "P", "*", "h", "H", "+", "x", "X",
        "D", "d", "|", "_", "none", " ",
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ] | list[tuple[int, int]] | tuple[int, Literal[0, 1, 2], int]
)
"""
Marker specification. See :doc:`/gallery/lines_bars_and_markers/marker_reference`.
"""

FillStyleType: TypeAlias = Literal["full", "left", "right", "bottom", "top", "none"]
"""Marker fill styles. See :doc:`/gallery/lines_bars_and_markers/marker_reference`."""

JoinStyleType: TypeAlias = JoinStyle | Literal["miter", "round", "bevel"]
"""Line join styles. See :doc:`/gallery/lines_bars_and_markers/joinstyle`."""

CapStyleType: TypeAlias = CapStyle | Literal["butt", "projecting", "round"]
"""Line cap styles. See :doc:`/gallery/lines_bars_and_markers/capstyle`."""

LogLevel: TypeAlias = Literal["notset", "debug", "info", "warning", "error", "critical"]
"""Literal type for valid logging levels accepted by `set_loglevel()`."""

CoordsBaseType = Union[
    str,
    Artist,
    Transform,
    Callable[
        [RendererBase],
        Union[Bbox, Transform]
    ]
]
CoordsType = Union[
    CoordsBaseType,
    tuple[CoordsBaseType, CoordsBaseType]
]

RcStyleType: TypeAlias = (
    str |
    dict[str, Any] |
    pathlib.Path |
    Sequence[str | pathlib.Path | dict[str, Any]]
)

_HT = TypeVar("_HT", bound=Hashable)
HashableList: TypeAlias = list[_HT | "HashableList[_HT]"]
"""A nested list of Hashable values."""

MouseEventType: TypeAlias = Literal[
    "button_press_event",
    "button_release_event",
    "motion_notify_event",
    "scroll_event",
    "figure_enter_event",
    "figure_leave_event",
    "axes_enter_event",
    "axes_leave_event",
]

KeyEventType: TypeAlias = Literal[
    "key_press_event",
    "key_release_event"
]

DrawEventType: TypeAlias = Literal["draw_event"]
PickEventType: TypeAlias = Literal["pick_event"]
ResizeEventType: TypeAlias = Literal["resize_event"]
CloseEventType: TypeAlias = Literal["close_event"]

EventType: TypeAlias = Literal[
    MouseEventType,
    KeyEventType,
    DrawEventType,
    PickEventType,
    ResizeEventType,
    CloseEventType,
]
