import io
import os
import pathlib

from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.axes import Axes
from matplotlib import cbook, cm
from matplotlib.backend_bases import FigureCanvasBase, RendererBase, MouseEvent
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.transforms import (
    Affine2D,
    Bbox,
    BboxBase,
    BboxTransform,
    BboxTransformTo,
    IdentityTransform,
    TransformedBbox,
    Transform,
)

from typing import Any, Literal, Sequence
import numpy as np
from numpy.typing import ArrayLike

import PIL  # type: ignore

interpolations_names: set[str]

def composite_images(
    images: Sequence[_ImageBase], renderer: RendererBase, magnification: float = ...
) -> tuple[np.ndarray, float, float]: ...

class _ImageBase(martist.Artist, cm.ScalarMappable):
    zorder: float
    origin: Literal["upper", "lower"]
    axes: Axes
    def __init__(
        self,
        ax: Axes,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        interpolation: str | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool | None = ...,
        *,
        interpolation_stage: Literal["data", "rgba"] | None = ...,
        **kwargs
    ) -> None: ...
    def get_size(self) -> tuple[int, int]: ...
    def set_alpha(self, alpha: float | ArrayLike | None) -> None: ...
    def changed(self) -> None: ...
    def make_image(
        self, renderer: RendererBase, magnification: float = ..., unsampled: bool = ...
    ) -> tuple[np.ndarray, float, float, Affine2D]: ...
    def draw(self, renderer: RendererBase, *args, **kwargs) -> None: ...
    def write_png(self, fname: str | pathlib.Path | io.FileIO) -> None: ...
    def set_data(self, A: ArrayLike | None) -> None: ...
    def set_array(self, A: ArrayLike | None) -> None: ...
    def get_interpolation(self) -> str: ...
    def set_interpolation(self, s: str) -> None: ...
    def set_interpolation_stage(self, s: Literal["data", "rgba"]) -> None: ...
    def can_composite(self) -> bool: ...
    def set_resample(self, v: bool | None) -> None: ...
    def get_resample(self) -> bool: ...
    def set_filternorm(self, filternorm: bool) -> None: ...
    def get_filternorm(self) -> bool: ...
    def set_filterrad(self, filterrad: float) -> None: ...
    def get_filterrad(self) -> float: ...

class AxesImage(_ImageBase):
    def __init__(
        self,
        ax: Axes,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        interpolation: str | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        extent: tuple[float, float, float, float] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool = ...,
        *,
        interpolation_stage: Literal["data", "rgba"] | None = ...,
        **kwargs
    ) -> None: ...
    def get_window_extent(self, renderer: RendererBase | None = ...): ...
    def make_image(
        self, renderer: RendererBase, magnification: float = ..., unsampled: bool = ...
    ): ...
    def set_extent(
        self, extent: tuple[float, float, float, float], **kwargs
    ) -> None: ...
    def get_extent(self) -> tuple[float, float, float, float]: ...
    def get_cursor_data(self, event: MouseEvent) -> None | float: ...

class NonUniformImage(AxesImage):
    mouseover: bool
    def __init__(
        self, ax: Axes, *, interpolation: Literal["nearest", "bilinear"] = ..., **kwargs
    ) -> None: ...
    def set_data(self, x: ArrayLike, y: ArrayLike, A: ArrayLike) -> None: ...  # type: ignore[override]
    # more limited interpolation available here than base class
    def set_interpolation(self, s: Literal["nearest", "bilinear"]) -> None: ...  # type: ignore[override]

class PcolorImage(AxesImage):
    def __init__(
        self,
        ax: Axes,
        x: ArrayLike | None = ...,
        y: ArrayLike | None = ...,
        A: ArrayLike | None = ...,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        **kwargs
    ) -> None: ...
    def set_data(self, x: ArrayLike, y: ArrayLike, A: ArrayLike) -> None: ...  # type: ignore[override]

class FigureImage(_ImageBase):
    zorder: float
    figure: Figure
    ox: float
    oy: float
    magnification: float
    def __init__(
        self,
        fig: Figure,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        offsetx: int = ...,
        offsety: int = ...,
        origin: Literal["upper", "lower"] | None = ...,
        **kwargs
    ) -> None: ...

class BboxImage(_ImageBase):
    bbox: Bbox
    def __init__(
        self,
        bbox: Bbox,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        interpolation: str | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool = ...,
        **kwargs
    ) -> None: ...
    def get_window_extent(self, renderer: RendererBase | None = ...): ...

def imread(fname: str | io.FileIO, format: str | None = ...) -> np.ndarray: ...
def imsave(
    fname: str | os.PathLike | io.FileIO,
    arr: ArrayLike,
    vmin: float | None = ...,
    vmax: float | None = ...,
    cmap: str | Colormap | None = ...,
    format: str | None = ...,
    origin: Literal["upper", "lower"] | None = ...,
    dpi: float = ...,
    *,
    metadata: dict[str, str] | None = ...,
    pil_kwargs: dict[str, Any] | None = ...
) -> None: ...
def pil_to_array(pilImage: PIL.Image.Image) -> np.ndarray: ...
def thumbnail(
    infile: str | io.FileIO,
    thumbfile: str | io.FileIO,
    scale: float = ...,
    interpolation: str = ...,
    preview: bool = ...,
) -> Figure: ...
