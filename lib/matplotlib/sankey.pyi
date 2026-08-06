from matplotlib.axes import Axes

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Self

import numpy as np

__license__: str
__credits__: list[str]
__author__: str
__version__: str

RIGHT: int
UP: int
DOWN: int

class Sankey:
    diagrams: list[SimpleNamespace]
    ax: Axes
    unit: str | None
    format: str | Callable[[float], str]
    scale: float
    gap: float
    radius: float
    shoulder: float
    offset: float
    margin: float
    pitch: float
    tolerance: float
    extent: np.ndarray
    def __init__(
        self,
        ax: Axes | None = ...,
        scale: float = ...,
        unit: str | None = ...,
        format: str | Callable[[float], str] = ...,
        gap: float = ...,
        radius: float = ...,
        shoulder: float = ...,
        offset: float = ...,
        head_angle: float = ...,
        margin: float = ...,
        tolerance: float = ...,
        **kwargs
    ) -> None: ...
    def add(
        self,
        patchlabel: str = ...,
        flows: Iterable[float] | None = ...,
        orientations: Iterable[int] | None = ...,
        labels: str | Iterable[str | None] = ...,
        trunklength: float = ...,
        pathlengths: float | Iterable[float] = ...,
        prior: int | None = ...,
        connect: tuple[int, int] = ...,
        rotation: float = ...,
        **kwargs
    ) -> Self: ...
    def finish(self) -> list[SimpleNamespace]: ...
