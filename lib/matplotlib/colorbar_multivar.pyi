from matplotlib import colorizer as mcolorizer
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import Bbox

import numpy as np
from collections.abc import Sequence
from typing import Any, Literal, overload

class MultivarColorbar(Sequence[Colorbar]):
    def __init__(
        self,
        axes: Sequence[Axes],
        mappable: mcolorizer.ColorizingArtist | mcolorizer.Colorizer,
        **kwargs: Any,
    ) -> None: ...
    @property
    def axes(self) -> list[Axes]: ...
    def update_normal(self, mappable: mcolorizer.ColorizingArtist | None = ...) -> None: ...
    def remove(self) -> None: ...
    @overload
    def __getitem__(self, index: int, /) -> Colorbar: ...
    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None], /) -> Sequence[Colorbar]: ...
    def __len__(self) -> int: ...
    def get_tightbbox(
        self, renderer: RendererBase | None = ..., for_layout_only: bool = ...
    ) -> Bbox: ...

def make_axes(
    parents: Axes | list[Axes] | np.ndarray,
    n_variates: int,
    n_major: int = ...,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    orientation: Literal["vertical", "horizontal"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    major_pad: float = ...,
    minor_pad: float = ...,
    **kwargs
) -> tuple[list[Axes], dict[str, Any], dict[str, Any]]: ...
