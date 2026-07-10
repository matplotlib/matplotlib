from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap

from collections.abc import Sequence
from numpy.typing import ArrayLike
from .typing import ColorType

def parallel_coordinates(
    axes: Axes,
    data: ArrayLike,
    class_column: str | int | None = ...,
    cols: Sequence[str | int] | None = ...,
    color: ColorType | Sequence[ColorType] | None = ...,
    cmap: str | Colormap | None = ...,
    alpha: float = ...,
    linewidth: float = ...,
    linestyle: str = ...,
) -> list[LineCollection]: ...

__all__ = ['parallel_coordinates']
