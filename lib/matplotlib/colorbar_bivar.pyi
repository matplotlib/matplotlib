from matplotlib import colorizer as mcolorizer
from matplotlib.axes import Axes
from matplotlib.axis import Axis

import numpy as np
from typing import Any, Literal

class BivarColorbar:
    def __init__(
        self,
        ax: Axes,
        mappable: mcolorizer.ColorizingArtist | mcolorizer.Colorizer,
        *,
        location: Literal["left", "right", "top", "bottom"] | None = ...,
        ticklocations: tuple[Literal["auto", "left", "right"], Literal["auto", "top", "bottom"]] = ...,
        aspect: float = ...,
    ) -> None: ...
    @property
    def ax(self) -> Axes: ...
    @property
    def aspect(self) -> float: ...
    @aspect.setter
    def aspect(self, aspect: float) -> None: ...
    def set_xlabel(self, label: str) -> None: ...
    def set_ylabel(self, label: str) -> None: ...
    @property
    def xaxis(self) -> Axis: ...
    @property
    def yaxis(self) -> Axis: ...
    def update_normal(self, mappable: mcolorizer.ColorizingArtist | None = ...) -> None: ...
    def remove(self) -> None: ...

def make_axes(
    parents: Axes | list[Axes] | np.ndarray,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
def make_axes_gridspec(
    parent: Axes,
    *,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
