from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict, Unpack

from numpy.typing import ArrayLike

from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.offsetbox import OffsetBox
from matplotlib.transforms import Transform

def update_from_first_child(tgt: Artist, src: Artist) -> None: ...

class _BaseKwargs(TypedDict, total=False):
    xpad: float
    ypad: float
    update_func: Callable[[Artist, Artist], None] | None

class HandlerBase:
    def __init__(
        self,
        xpad: float = ...,
        ypad: float = ...,
        update_func: Callable[[Artist, Artist], None] | None = ...,
    ) -> None: ...
    def update_prop(
        self, legend_handle: Artist, orig_handle: Artist, legend: Legend
    ) -> None: ...
    def adjust_drawing_area(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[float, float, float, float]: ...
    def legend_artist(
        self, legend: Legend, orig_handle: Artist, fontsize: float, handlebox: OffsetBox
    ) -> Artist: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerNpoints(HandlerBase):
    def __init__(
        self, marker_pad: float = ..., numpoints: int | None = ..., **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def get_numpoints(self, legend: Legend) -> int | None: ...
    def get_xdata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[ArrayLike, ArrayLike]: ...

class HandlerNpointsYoffsets(HandlerNpoints):
    def __init__(
        self,
        numpoints: int | None = ...,
        yoffsets: Sequence[float] | None = ...,
        *,
        # From HandlerNpoints
        marker_pad: float = ...,
        **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def get_ydata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> ArrayLike: ...

class HandlerLine2DCompound(HandlerNpoints):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerLine2D(HandlerNpoints):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class _PatchFunc(Protocol):
    def __call__(
        self,
        *,
        legend: Legend = ...,
        orig_handle: Artist = ...,
        xdescent: float = ...,
        ydescent: float = ...,
        width: float = ...,
        height: float = ...,
        fontsize: float = ...,
    ) -> Artist: ...

class HandlerPatch(HandlerBase):
    def __init__(self, patch_func: _PatchFunc | None = ..., **kwargs: Unpack[_BaseKwargs]) -> None: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerStepPatch(HandlerBase):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerLineCollection(HandlerLine2D):
    def get_numpoints(self, legend: Legend) -> int: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    def __init__(
        self,
        yoffsets: Sequence[float] | None = ...,
        sizes: Sequence[float] | None = ...,
        *,
        # From HandlerNpoints
        marker_pad: float = ...,
        numpoints: int | None = ...,
        **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def get_numpoints(self, legend: Legend) -> int: ...
    def get_sizes(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> Sequence[float]: ...
    def update_prop(
        self, legend_handle: Artist, orig_handle: Artist, legend: Legend
    ) -> None: ...
    def create_collection[T: Artist](
        self,
        orig_handle: T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> T: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerPathCollection(HandlerRegularPolyCollection):
    def create_collection[T: Artist](
        self,
        orig_handle: T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> T: ...

class HandlerCircleCollection(HandlerRegularPolyCollection):
    def create_collection[T: Artist](
        self,
        orig_handle: T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> T: ...

class HandlerErrorbar(HandlerLine2D):
    def __init__(
        self,
        xerr_size: float = ...,
        yerr_size: float | None = ...,
        marker_pad: float = ...,
        numpoints: int | None = ...,
        **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def get_err_size(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[float, float]: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerStem(HandlerNpointsYoffsets):
    def __init__(
        self,
        marker_pad: float = ...,
        numpoints: int | None = ...,
        bottom: float | None = ...,
        yoffsets: Sequence[float] | None = ...,
        **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def get_ydata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> ArrayLike: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerTuple(HandlerBase):
    def __init__(
        self, ndivide: int | None = ..., pad: float | None = ..., **kwargs: Unpack[_BaseKwargs]
    ) -> None: ...
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

class HandlerPolyCollection(HandlerBase):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
