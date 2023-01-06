from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from typing import Any, Callable, Iterable

import numpy as np

RIGHT: int
UP: int
DOWN: int

# TODO typing units
class Sankey:
    diagrams: list[Any]
    ax: Axes
    unit: Any
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
        unit: Any = ...,
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
    ): ...
    def finish(self) -> list[Any]: ...
