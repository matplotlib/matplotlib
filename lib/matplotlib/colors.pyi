from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from matplotlib import cbook, scale
import re

from typing import Any, Literal, overload
from .typing import ColorType

import numpy as np
from numpy.typing import ArrayLike

# Explicitly export colors dictionaries which are imported in the impl
BASE_COLORS: dict[str, ColorType]
CSS4_COLORS: dict[str, ColorType]
TABLEAU_COLORS: dict[str, ColorType]
XKCD_COLORS: dict[str, ColorType]
SPECTRAL_COLORS: dict[str, ColorType]

class _ColorMapping(dict[str, ColorType]):
    cache: dict[tuple[ColorType, float | None], tuple[float, float, float, float]]
    def __init__(self, mapping) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...

def get_named_colors_mapping() -> _ColorMapping: ...

class ColorSequenceRegistry(Mapping):
    def __init__(self) -> None: ...
    def __getitem__(self, item: str) -> list[ColorType]: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def register(self, name: str, color_list: Iterable[ColorType]) -> None: ...
    def unregister(self, name: str) -> None: ...

_color_sequences: ColorSequenceRegistry = ...

def is_color_like(c: Any) -> bool: ...
def same_color(c1: ColorType, c2: ColorType) -> bool: ...
def to_rgba(
    c: ColorType, alpha: float | None = ...
) -> tuple[float, float, float, float]: ...
def to_rgba_array(
    c: ColorType | ArrayLike, alpha: float | ArrayLike | None = ...
) -> np.ndarray: ...
def to_rgb(c: ColorType) -> tuple[float, float, float]: ...
def to_hex(c: ColorType, keep_alpha: bool = ...) -> str: ...

cnames: dict[str, ColorType]
hexColorPattern: re.Pattern
rgb2hex = to_hex
hex2color = to_rgb

class ColorConverter:
    colors: _ColorMapping
    cache: dict[tuple[ColorType, float | None], tuple[float, float, float, float]]
    @staticmethod
    def to_rgb(c: ColorType) -> tuple[float, float, float]: ...
    @staticmethod
    def to_rgba(
        c: ColorType, alpha: float | None = ...
    ) -> tuple[float, float, float, float]: ...
    @staticmethod
    def to_rgba_array(
        c: ColorType | ArrayLike, alpha: float | ArrayLike | None = ...
    ) -> np.ndarray: ...

colorConverter: ColorConverter

class Colormap:
    name: str
    N: int
    colorbar_extend: bool
    def __init__(
            self,
            name: str,
            N: int = ...,
            *,
            bad: ColorType | None = ...,
            under: ColorType | None = ...,
            over: ColorType | None = ...
    ) -> None: ...
    @overload
    def __call__(
        self, X: Sequence[float] | np.ndarray, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> np.ndarray: ...
    @overload
    def __call__(
        self, X: float, alpha: float | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float]: ...
    @overload
    def __call__(
        self, X: ArrayLike, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float] | np.ndarray: ...
    def __copy__(self) -> Colormap: ...
    def __eq__(self, other: object) -> bool: ...
    def get_bad(self) -> np.ndarray: ...
    def set_bad(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    def get_under(self) -> np.ndarray: ...
    def set_under(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    def get_over(self) -> np.ndarray: ...
    def set_over(self, color: ColorType = ..., alpha: float | None = ...) -> None: ...
    def set_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        under: ColorType | None = ...,
        over: ColorType | None = ...
    ) -> None: ...
    def with_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        under: ColorType | None = ...,
        over: ColorType | None = ...
    ) -> Colormap: ...
    def with_alpha(self, alpha: float) -> Colormap: ...
    def is_gray(self) -> bool: ...
    def resampled(self, lutsize: int) -> Colormap: ...
    def reversed(self, name: str | None = ...) -> Colormap: ...
    def _repr_html_(self) -> str: ...
    def _repr_png_(self) -> bytes: ...
    def copy(self) -> Colormap: ...

class LinearSegmentedColormap(Colormap):
    monochrome: bool
    def __init__(
        self,
        name: str,
        segmentdata: dict[
            Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]
        ],
        N: int = ...,
        gamma: float = ...,
        *,
        bad: ColorType | None = ...,
        under: ColorType | None = ...,
        over: ColorType | None = ...,
    ) -> None: ...
    def set_gamma(self, gamma: float) -> None: ...
    @staticmethod
    def from_list(
        name: str, colors: ArrayLike | Sequence[tuple[float, ColorType]], N: int = ..., gamma: float = ...,
        *, bad: ColorType | None = ..., under: ColorType | None = ..., over: ColorType | None = ...,
    ) -> LinearSegmentedColormap: ...
    def resampled(self, lutsize: int) -> LinearSegmentedColormap: ...
    def reversed(self, name: str | None = ...) -> LinearSegmentedColormap: ...

class ListedColormap(Colormap):
    colors: ArrayLike | ColorType
    def __init__(
        self, colors: ArrayLike | ColorType, name: str = ..., N: int | None = ...,
        *, bad: ColorType | None = ..., under: ColorType | None = ..., over: ColorType | None = ...
    ) -> None: ...
    @property
    def monochrome(self) -> bool: ...
    def resampled(self, lutsize: int) -> ListedColormap: ...
    def reversed(self, name: str | None = ...) -> ListedColormap: ...

class MultivarColormap:
    name: str
    n_variates: int
    def __init__(self, colormaps: list[Colormap], combination_mode: Literal['sRGB_add', 'sRGB_sub'], name: str = ...) -> None: ...
    @overload
    def __call__(
        self, X: Sequence[Sequence[float]] | np.ndarray, alpha: ArrayLike | None = ..., bytes: bool = ..., clip: bool = ...
    ) -> np.ndarray: ...
    @overload
    def __call__(
        self, X: Sequence[float], alpha: float | None = ..., bytes: bool = ..., clip: bool = ...
    ) -> tuple[float, float, float, float]: ...
    @overload
    def __call__(
        self, X: ArrayLike, alpha: ArrayLike | None = ..., bytes: bool = ..., clip: bool = ...
    ) -> tuple[float, float, float, float] | np.ndarray: ...
    def copy(self) -> MultivarColormap: ...
    def __copy__(self) -> MultivarColormap: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getitem__(self, item: int) -> Colormap: ...
    def __iter__(self) -> Iterator[Colormap]: ...
    def __len__(self) -> int: ...
    def get_bad(self) -> np.ndarray: ...
    def resampled(self, lutshape: Sequence[int | None]) -> MultivarColormap: ...
    def with_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        under: Sequence[ColorType] | None = ...,
        over: Sequence[ColorType] | None = ...
    ) -> MultivarColormap: ...
    @property
    def combination_mode(self) -> str: ...
    def _repr_html_(self) -> str: ...
    def _repr_png_(self) -> bytes: ...

class BivarColormap:
    name: str
    N: int
    M: int
    n_variates: int
    def __init__(
    	self, N: int = ..., M: int | None = ..., shape: Literal['square', 'circle', 'ignore', 'circleignore'] = ...,
    	origin: Sequence[float] = ..., name: str = ...
    ) -> None: ...
    @overload
    def __call__(
        self, X: Sequence[Sequence[float]] | np.ndarray, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> np.ndarray: ...
    @overload
    def __call__(
        self, X: Sequence[float], alpha: float | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float]: ...
    @overload
    def __call__(
        self, X: ArrayLike, alpha: ArrayLike | None = ..., bytes: bool = ...
    ) -> tuple[float, float, float, float] | np.ndarray: ...
    @property
    def lut(self) -> np.ndarray: ...
    @property
    def shape(self) -> str: ...
    @property
    def origin(self) -> tuple[float, float]: ...
    def copy(self) -> BivarColormap: ...
    def __copy__(self) -> BivarColormap: ...
    def __getitem__(self, item: int) -> Colormap: ...
    def __eq__(self, other: Any) -> bool: ...
    def get_bad(self) -> np.ndarray: ...
    def get_outside(self) -> np.ndarray: ...
    def resampled(self, lutshape: Sequence[int | None], transposed: bool = ...) -> BivarColormap: ...
    def transposed(self) -> BivarColormap: ...
    def reversed(self, axis_0: bool = ..., axis_1: bool = ...) -> BivarColormap: ...
    def with_extremes(
        self,
        *,
        bad: ColorType | None = ...,
        outside: ColorType | None = ...,
        shape: str | None = ...,
        origin: None | Sequence[float] = ...,
    ) -> MultivarColormap: ...
    def _repr_html_(self) -> str: ...
    def _repr_png_(self) -> bytes: ...

class SegmentedBivarColormap(BivarColormap):
    def __init__(
        self, patch: np.ndarray, N: int = ..., shape: Literal['square', 'circle', 'ignore', 'circleignore'] = ...,
        origin: Sequence[float] = ..., name: str = ...
    ) -> None: ...

class BivarColormapFromImage(BivarColormap):
    def __init__(
    	self, lut: np.ndarray, shape: Literal['square', 'circle', 'ignore', 'circleignore'] = ...,
    	origin: Sequence[float] = ..., name: str = ...
    ) -> None: ...

class Normalize:
    callbacks: cbook.CallbackRegistry
    def __init__(
        self, vmin: float | None = ..., vmax: float | None = ..., clip: bool = ...
    ) -> None: ...
    @property
    def vmin(self) -> float | None: ...
    @vmin.setter
    def vmin(self, value: float | None) -> None: ...
    @property
    def vmax(self) -> float | None: ...
    @vmax.setter
    def vmax(self, value: float | None) -> None: ...
    @property
    def clip(self) -> bool: ...
    @clip.setter
    def clip(self, value: bool) -> None: ...
    @staticmethod
    def process_value(value: ArrayLike) -> tuple[np.ma.MaskedArray, bool]: ...
    @overload
    def __call__(self, value: float, clip: bool | None = ...) -> float: ...
    @overload
    def __call__(self, value: np.ndarray, clip: bool | None = ...) -> np.ma.MaskedArray: ...
    @overload
    def __call__(self, value: ArrayLike, clip: bool | None = ...) -> ArrayLike: ...
    @overload
    def inverse(self, value: float) -> float: ...
    @overload
    def inverse(self, value: np.ndarray) -> np.ma.MaskedArray: ...
    @overload
    def inverse(self, value: ArrayLike) -> ArrayLike: ...
    def autoscale(self, A: ArrayLike) -> None: ...
    def autoscale_None(self, A: ArrayLike) -> None: ...
    def scaled(self) -> bool: ...

class TwoSlopeNorm(Normalize):
    def __init__(
        self, vcenter: float, vmin: float | None = ..., vmax: float | None = ...
    ) -> None: ...
    @property
    def vcenter(self) -> float: ...
    @vcenter.setter
    def vcenter(self, value: float) -> None: ...
    def autoscale_None(self, A: ArrayLike) -> None: ...

class CenteredNorm(Normalize):
    def __init__(
        self, vcenter: float = ..., halfrange: float | None = ..., clip: bool = ...
    ) -> None: ...
    @property
    def vcenter(self) -> float: ...
    @vcenter.setter
    def vcenter(self, vcenter: float) -> None: ...
    @property
    def halfrange(self) -> float: ...
    @halfrange.setter
    def halfrange(self, halfrange: float) -> None: ...

@overload
def make_norm_from_scale(
    scale_cls: type[scale.ScaleBase],
    base_norm_cls: type[Normalize],
    *,
    init: Callable | None = ...
) -> type[Normalize]: ...
@overload
def make_norm_from_scale(
    scale_cls: type[scale.ScaleBase],
    base_norm_cls: None = ...,
    *,
    init: Callable | None = ...
) -> Callable[[type[Normalize]], type[Normalize]]: ...

class FuncNorm(Normalize):
    def __init__(
            self,
            functions: tuple[Callable, Callable],
            vmin: float | None = ...,
            vmax: float | None = ...,
            clip: bool = ...,
    ) -> None: ...
class LogNorm(Normalize): ...

class SymLogNorm(Normalize):
    def __init__(
            self,
            linthresh: float,
            linscale: float = ...,
            vmin: float | None = ...,
            vmax: float | None = ...,
            clip: bool = ...,
            *,
            base: float = ...,
    ) -> None: ...
    @property
    def linthresh(self) -> float: ...
    @linthresh.setter
    def linthresh(self, value: float) -> None: ...

class AsinhNorm(Normalize):
    def __init__(
        self,
        linear_width: float = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        clip: bool = ...,
    ) -> None: ...
    @property
    def linear_width(self) -> float: ...
    @linear_width.setter
    def linear_width(self, value: float) -> None: ...

class PowerNorm(Normalize):
    gamma: float
    def __init__(
        self,
        gamma: float,
        vmin: float | None = ...,
        vmax: float | None = ...,
        clip: bool = ...,
    ) -> None: ...

class BoundaryNorm(Normalize):
    boundaries: np.ndarray
    N: int
    Ncmap: int
    extend: Literal["neither", "both", "min", "max"]
    def __init__(
        self,
        boundaries: ArrayLike,
        ncolors: int,
        clip: bool = ...,
        *,
        extend: Literal["neither", "both", "min", "max"] = ...
    ) -> None: ...

class NoNorm(Normalize): ...

def rgb_to_hsv(arr: ArrayLike) -> np.ndarray: ...
def hsv_to_rgb(hsv: ArrayLike) -> np.ndarray: ...

class LightSource:
    azdeg: float
    altdeg: float
    hsv_min_val: float
    hsv_max_val: float
    hsv_min_sat: float
    hsv_max_sat: float
    def __init__(
        self,
        azdeg: float = ...,
        altdeg: float = ...,
        hsv_min_val: float = ...,
        hsv_max_val: float = ...,
        hsv_min_sat: float = ...,
        hsv_max_sat: float = ...,
    ) -> None: ...
    @property
    def direction(self) -> np.ndarray: ...
    def hillshade(
        self,
        elevation: ArrayLike,
        vert_exag: float = ...,
        dx: float = ...,
        dy: float = ...,
        fraction: float = ...,
    ) -> np.ndarray: ...
    def shade_normals(
        self, normals: np.ndarray, fraction: float = ...
    ) -> np.ndarray: ...
    def shade(
        self,
        data: ArrayLike,
        cmap: Colormap,
        norm: Normalize | None = ...,
        blend_mode: Literal["hsv", "overlay", "soft"] | Callable = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        vert_exag: float = ...,
        dx: float = ...,
        dy: float = ...,
        fraction: float = ...,
        **kwargs
    ) -> np.ndarray: ...
    def shade_rgb(
        self,
        rgb: ArrayLike,
        elevation: ArrayLike,
        fraction: float = ...,
        blend_mode: Literal["hsv", "overlay", "soft"] | Callable = ...,
        vert_exag: float = ...,
        dx: float = ...,
        dy: float = ...,
        **kwargs
    ) -> np.ndarray: ...
    def blend_hsv(
        self,
        rgb: ArrayLike,
        intensity: ArrayLike,
        hsv_max_sat: float | None = ...,
        hsv_max_val: float | None = ...,
        hsv_min_val: float | None = ...,
        hsv_min_sat: float | None = ...,
    ) -> ArrayLike: ...
    def blend_soft_light(
        self, rgb: np.ndarray, intensity: np.ndarray
    ) -> np.ndarray: ...
    def blend_overlay(self, rgb: np.ndarray, intensity: np.ndarray) -> np.ndarray: ...

def from_levels_and_colors(
    levels: Sequence[float],
    colors: Sequence[ColorType],
    extend: Literal["neither", "min", "max", "both"] = ...,
) -> tuple[ListedColormap, BoundaryNorm]: ...
