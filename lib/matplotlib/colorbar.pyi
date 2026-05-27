import matplotlib.spines as mspines
from matplotlib import cm, collections, colors, contour, colorizer as mcolorizer
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backend_bases import RendererBase
from matplotlib.patches import Patch
from matplotlib.ticker import Locator, Formatter
from matplotlib.transforms import Bbox

import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
from typing import Any, Literal, overload
from .typing import ColorType

class _ColorbarSpine(mspines.Spines):
    def __init__(self, axes: Axes): ...
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:...
    def set_xy(self, xy: ArrayLike) -> None: ...
    def draw(self, renderer: RendererBase | None) -> None:...


class Colorbar:
    n_rasterize: int
    mappable: cm.ScalarMappable | mcolorizer.ColorizingArtist
    ax: Axes
    alpha: float | None
    cmap: colors.Colormap
    norm: colors.Normalize
    values: Sequence[float] | None
    boundaries: Sequence[float] | None
    extend: Literal["neither", "both", "min", "max"]
    spacing: Literal["uniform", "proportional"]
    orientation: Literal["vertical", "horizontal"]
    drawedges: bool
    extendfrac: Literal["auto"] | float | Sequence[float] | None
    extendrect: bool
    solids: None | collections.QuadMesh
    solids_patches: list[Patch]
    lines: list[collections.LineCollection]
    outline: _ColorbarSpine
    dividers: collections.LineCollection
    ticklocation: Literal["left", "right", "top", "bottom"]
    def __init__(
        self,
        ax: Axes,
        mappable: cm.ScalarMappable | mcolorizer.ColorizingArtist | None = ...,
        *,
        cmap: str | colors.Colormap | None = ...,
        norm: colors.Normalize | None = ...,
        alpha: float | None = ...,
        values: Sequence[float] | None = ...,
        boundaries: Sequence[float] | None = ...,
        orientation: Literal["vertical", "horizontal"] | None = ...,
        ticklocation: Literal["auto", "left", "right", "top", "bottom"] = ...,
        extend: Literal["neither", "both", "min", "max"] | None = ...,
        spacing: Literal["uniform", "proportional"] = ...,
        ticks: Sequence[float] | Locator | None = ...,
        format: str | Formatter | None = ...,
        drawedges: bool = ...,
        extendfrac: Literal["auto"] | float | Sequence[float] | None = ...,
        extendrect: bool = ...,
        label: str = ...,
        location: Literal["left", "right", "top", "bottom"] | None = ...
    ) -> None: ...
    @property
    def long_axis(self) -> Axis: ...
    @property
    def locator(self) -> Locator: ...
    @locator.setter
    def locator(self, loc: Locator) -> None: ...
    @property
    def minorlocator(self) -> Locator: ...
    @minorlocator.setter
    def minorlocator(self, loc: Locator) -> None: ...
    @property
    def formatter(self) -> Formatter: ...
    @formatter.setter
    def formatter(self, fmt: Formatter) -> None: ...
    @property
    def minorformatter(self) -> Formatter: ...
    @minorformatter.setter
    def minorformatter(self, fmt: Formatter) -> None: ...
    def update_normal(self, mappable: cm.ScalarMappable | None = ...) -> None: ...
    @overload
    def add_lines(self, CS: contour.ContourSet, erase: bool = ...) -> None: ...
    @overload
    def add_lines(
        self,
        levels: ArrayLike,
        colors: ColorType | Sequence[ColorType],
        linewidths: float | ArrayLike,
        erase: bool = ...,
    ) -> None: ...
    def update_ticks(self) -> None: ...
    def set_ticks(
        self,
        ticks: Sequence[float] | Locator,
        *,
        labels: Sequence[str] | None = ...,
        minor: bool = ...,
        **kwargs
    ) -> None: ...
    def get_ticks(self, minor: bool = ...) -> np.ndarray: ...
    def set_ticklabels(
        self,
        ticklabels: Sequence[str],
        *,
        minor: bool = ...,
        **kwargs
    ) -> None: ...
    def minorticks_on(self) -> None: ...
    def minorticks_off(self) -> None: ...
    def set_label(self, label: str, *, loc: str | None = ..., **kwargs) -> None: ...
    def set_alpha(self, alpha: float | np.ndarray) -> None: ...
    def remove(self) -> None: ...
    def drag_pan(self, button: Any, key: Any, x: float, y: float) -> None: ...


ColorbarBase = Colorbar

class BivarColorbar:
    n_rasterize: int
    mappable: mcolorizer.ColorizingArtist
    ax: Axes
    alpha: float | None
    colorizer: mcolorizer.Colorizer
    ticklocations: tuple[Literal["auto", "left", "right"], Literal["auto", "top", "bottom"]]
    def __init__(
        self,
        ax: Axes,
        mappable: mcolorizer.ColorizingArtist | mcolorizer.Colorizer,
        *,
        alpha: float | None = ...,
        location: Literal["left", "right", "top", "bottom"] | None = ...,
        ticklocations: tuple[Literal["auto", "left", "right"], Literal["auto", "top", "bottom"]] = ...,
        aspect: float = ...,
    ) -> None: ...
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
    def update_normals(self, mappable: mcolorizer.ColorizingArtist | None = ...) -> None: ...
    def set_alpha(self, alpha: float | None) -> None: ...
    def remove(self) -> None: ...
    def drag_pan(self, button: Any, key: Any, x: float, y: float) -> None: ...

class MultivarColorbar(Sequence[Colorbar]):
    mappable: mcolorizer.ColorizingArtist
    colorizer: mcolorizer.Colorizer
    axes: Sequence[Axes]
    _colorbars: list[Colorbar]
    def __init__(
        self,
        axes: Sequence[Axes],
        mappable: mcolorizer.ColorizingArtist | mcolorizer.Colorizer | None = ...,
        **kwargs: Any,
    ) -> None: ...

    def update_normals(self, mappable: mcolorizer.ColorizingArtist | None = ...) -> None: ...
    def remove(self) -> None: ...
    @overload
    def __getitem__(self, index: int, /) -> Colorbar: ...
    @overload
    def __getitem__(self, index: slice[int | None, int | None, int | None], /) -> Sequence[Colorbar]: ...
    def __len__(self) -> int: ...
    def get_tightbbox(self, renderer: RendererBase | None = ..., for_layout_only: bool = ...) -> Bbox: ...

def make_axes(
    parents: Axes | list[Axes] | np.ndarray,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    orientation: Literal["vertical", "horizontal"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
def make_bivar_axes(
    parents: Axes | list[Axes] | np.ndarray,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
def make_multivar_axes(
    parents: Axes | list[Axes] | np.ndarray,
    n_variates: int,
    n_major: int,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    orientation: Literal["vertical", "horizontal"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    major_pad: float = ...,
    minor_pad: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any], dict[str, Any]]: ...
def make_axes_gridspec(
    parent: Axes,
    *,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    orientation: Literal["vertical", "horizontal"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
def make_bivar_axes_gridspec(
    parent: Axes | list[Axes] | np.ndarray,
    *,
    location: Literal["left", "right", "top", "bottom"] | None = ...,
    fraction: float = ...,
    shrink: float = ...,
    aspect: float = ...,
    **kwargs
) -> tuple[Axes, dict[str, Any]]: ...
