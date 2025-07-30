"""
Plotting of string "category" data: ``plot(['d', 'f', 'a'], [1, 2, 3])`` will
plot three points with x-axis values of 'd', 'f', 'a'.

See :doc:`/gallery/lines_bars_and_markers/categorical_variables` for an
example.

The module uses Matplotlib's `matplotlib.units` mechanism to convert from
strings to integers and provides a tick locator, a tick formatter, and the
`.UnitData` class that creates and stores the string-to-integer mapping.
"""

from __future__ import annotations
from matplotlib.units import ConversionInterface, AxisInfo
from matplotlib import ticker
from matplotlib.axis import Axis
from collections import OrderedDict
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Union

class StrCategoryConverter(ConversionInterface):
    @staticmethod
    def convert(
        value: Union[str, Iterable[str]], 
        unit: UnitData, 
        axis: Axis
    ) -> Union[float, np.ndarray]: ...
    @staticmethod
    def axisinfo(unit: UnitData, axis: Axis) -> AxisInfo: ...
    @staticmethod
    def default_units(
        data: Union[str, Iterable[str]], 
        axis: Axis
    ) -> UnitData: ...
    @staticmethod
    def _validate_unit(unit: UnitData) -> None: ...

class StrCategoryLocator(ticker.Locator):
    def __init__(self, units_mapping: Dict[str, int]) -> None: ...
    def __call__(self) -> List[int]: ...
    def tick_values(self, vmin: Any, vmax: Any) -> List[int]: ...

class StrCategoryFormatter(ticker.Formatter):
    def __init__(self, units_mapping: Dict[str, int]) -> None: ...
    def __call__(self, x: float, pos: Optional[int] = None) -> str: ...
    def format_ticks(self, values: Iterable[float]) -> List[str]: ...
    @staticmethod
    def _text(value: Union[str, bytes]) -> str: ...

class UnitData:
    _mapping: OrderedDict[Union[str, bytes], int]
    def __init__(
        self, 
        data: Optional[Iterable[Union[str, bytes]]] = None
    ) -> None: ...
    def update(self, data: Iterable[Union[str, bytes]]) -> None: ...
    @staticmethod
    def _str_is_convertible(val: Union[str, bytes]) -> bool: ...
