# Stubs for the _backend_agg C-extension (src/_backend_agg_wrapper.cpp).
#
# Parameter names match the ``py::arg`` names used in the pybind11 bindings.
# The extension accepts any object convertible by the matplotlib type casters
# (``py_converters.h``); the annotations below name the objects that the
# Python-level ``backend_agg.RendererAgg`` actually passes.

from collections.abc import Buffer, Sequence
from typing import Any, overload

from numpy.typing import ArrayLike

from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.path import Path
from matplotlib.transforms import BboxBase, Transform

from matplotlib.typing import ColorType

class BufferRegion(Buffer):
    # Not constructible from Python; instances come from
    # ``RendererAgg.copy_from_bbox``.
    def set_x(self, x: int) -> None: ...
    def set_y(self, y: int) -> None: ...
    def get_extents(self) -> tuple[int, int, int, int]: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...

class RendererAgg(Buffer):
    def __init__(self, width: int, height: int, dpi: float) -> None: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        path: Path,
        trans: Transform,
        face: ColorType | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path: Path,
        marker_path_trans: Transform,
        path: Path,
        trans: Transform,
        face: ColorType | None = ...,
    ) -> None: ...
    def draw_text_image(
        self,
        image: ArrayLike,
        x: int,
        y: int,
        angle: float,
        gc: GraphicsContextBase,
    ) -> None: ...
    def draw_image(
        self, gc: GraphicsContextBase, x: float, y: float, image: ArrayLike
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[Path],
        transforms: ArrayLike,
        offsets: ArrayLike,
        offset_trans: Transform,
        facecolors: ArrayLike,
        edgecolors: ArrayLike,
        linewidths: ArrayLike,
        dashes: Sequence[tuple[float, Sequence[float] | None]],
        antialiaseds: ArrayLike,
        ignored: Any,
        offset_position: Any,
        *,
        hatchcolors: ArrayLike = ...,
    ) -> None: ...
    def draw_quad_mesh(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        mesh_width: int,
        mesh_height: int,
        coordinates: ArrayLike,
        offsets: ArrayLike,
        offset_trans: Transform,
        facecolors: ArrayLike,
        antialiased: bool,
        edgecolors: ArrayLike,
    ) -> None: ...
    def draw_gouraud_triangles(
        self,
        gc: GraphicsContextBase,
        points: ArrayLike,
        colors: ArrayLike,
        trans: Transform | None = ...,
    ) -> None: ...
    def clear(self) -> None: ...
    # ``bbox`` may be anything convertible to a 4-vector or 2x2 array
    # (``agg::rect_d`` caster), including any ``BboxBase``.
    def copy_from_bbox(self, bbox: BboxBase | ArrayLike) -> BufferRegion: ...
    @overload
    def restore_region(self, region: BufferRegion) -> None: ...
    @overload
    def restore_region(
        self,
        region: BufferRegion,
        xx1: int,
        yy1: int,
        xx2: int,
        yy2: int,
        x: int,
        y: int,
    ) -> None: ...
