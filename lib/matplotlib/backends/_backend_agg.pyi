# Stub generated from the C++ (pybind11) signatures in
# ``src/_backend_agg_wrapper.cpp``. NOT verified with matplotlib's ``stubtest``
# (the extension was not importable when this stub was written); the
# graphics-primitive argument types in particular are best-effort.
from collections.abc import Buffer, Sequence
from typing import overload

import numpy as np

from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.path import Path
from matplotlib.transforms import Bbox, Transform

class BufferRegion(Buffer):
    # Not constructible from Python.
    def set_x(self, x: int) -> None: ...
    def set_y(self, y: int) -> None: ...
    def get_extents(self) -> tuple[int, int, int, int]: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...

class RendererAgg(Buffer):
    def __init__(self, width: int, height: int, dpi: float) -> None: ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        path: Path,
        trans: Transform,
        face: np.ndarray | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path: Path,
        marker_path_trans: Transform,
        path: Path,
        trans: Transform,
        face: np.ndarray | None = ...,
    ) -> None: ...
    def draw_text_image(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        angle: float,
        gc: GraphicsContextBase,
    ) -> None: ...
    def draw_image(
        self, gc: GraphicsContextBase, x: float, y: float, image: np.ndarray
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[Path],
        transforms: Sequence[np.ndarray],
        offsets: np.ndarray,
        offset_trans: Transform,
        facecolors: np.ndarray,
        edgecolors: np.ndarray,
        linewidths: np.ndarray,
        dashes: Sequence,
        antialiaseds: np.ndarray,
        ignored: object,
        offset_position: object,
        *,
        hatchcolors: np.ndarray = ...,
    ) -> None: ...
    def draw_quad_mesh(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        mesh_width: int,
        mesh_height: int,
        coordinates: np.ndarray,
        offsets: np.ndarray,
        offset_trans: Transform,
        facecolors: np.ndarray,
        antialiased: bool,
        edgecolors: np.ndarray,
    ) -> None: ...
    def draw_gouraud_triangles(
        self,
        gc: GraphicsContextBase,
        points: np.ndarray,
        colors: np.ndarray,
        trans: Transform | None = ...,
    ) -> None: ...
    def clear(self) -> None: ...
    def copy_from_bbox(self, bbox: Bbox) -> BufferRegion: ...
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
    def __buffer__(self, flags: int, /) -> memoryview: ...
