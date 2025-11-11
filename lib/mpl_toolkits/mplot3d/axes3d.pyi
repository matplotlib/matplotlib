from __future__ import annotations

__all__ = ["Axes3D", "_Quaternion", "get_test_data"]

from typing import Literal, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.colors import LightSource, Normalize
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from matplotlib.contour import QuadContourSet
from matplotlib.collections import Collection, PathCollection
from matplotlib.transforms import Bbox
from matplotlib.backend_bases import RendererBase, MouseButton
from matplotlib.tri.tricontour import TriContourSet
from axis3d import ZAxis
from art3d import Poly3DCollection, Line3DCollection

Number = int | float | np.ndarray
ArrayLike = npt.ArrayLike


class Axes3D(Axes):
    name: str
    initial_azim: float
    initial_elev: float
    initial_roll: float
    computed_zorder: bool
    M: np.ndarray | None
    invM: np.ndarray | None

    def __init__(
        self,
        fig: Figure,
        rect: tuple[float, float, float, float] | None,
        elev: float,
        azim: float,
        roll: float,
        shareview: "Axes3D" | None,
        sharez: "Axes3D" | None,
        proj_type: str,
        focal_length: float | None,
        box_aspect: Sequence[float] | None,
        computed_zorder: bool,
    ) -> None: ...

    def set_axis_off(self) -> None: ...

    def set_axis_on(self) -> None: ...

    def convert_zunits(self, z): ...

    def set_top_view(self) -> None: ...

    def _init_axis(self) -> None: ...

    def get_zaxis(self) -> ZAxis: ...

    def _transformed_cube(
        self,
        vals: tuple[float, float, float, float, float, float]
    ) -> ArrayLike: ...

    def set_aspect(
        self,
        aspect: float | str,
        adjustable: None | Literal['box'] | Literal['datalim']
    ) -> None:
        ...

    def _equal_aspect_axis_indices(self, aspect: str) -> None: ...

    def set_box_aspect(
        self,
        aspect: Sequence[float] | None,
        zoom: float
    ) -> None: ...

    def apply_aspect(self, position: Bbox | None) -> None: ...

    def draw(self, renderer: RendererBase) -> None: ...

    def get_axis_position(self) -> tuple: ...

    def set_zmargin(self, m: float): ...

    def margins(
        self,
        margins,
        x, y, z,
        tight: bool
    ) -> None | tuple: ...

    def autoscale(
        self,
        enable: bool,
        axis: str,
        tight: bool | None
    ) -> None: ...

    def auto_scale_xyz(
        self,
        X: ArrayLike, Y:ArrayLike, Z:ArrayLike,
        had_data: bool
    ) -> None: ...

    def autoscale_view(
            self,
            tight: bool | None,
            scalex: bool,
            scaley: bool,
            scalez: bool
        ) -> None: ...

    def get_w_lims(self) -> tuple[
        float, float, float, float, float, float]: ...

    def _set_bound3d(
        self,
        get_bound: tuple,
        set_lim,
        axis_inverted,
        lower,
        upper,
        view_margin
        ) -> None: ...

    def set_xbound(
        self,
        lower: float | None,
        upper: float | None,
        view_margin: float | None
    ) -> None: ...

    def set_ybound(
        self,
        lower: float | None,
        upper: float | None,
        view_margin: float | None
    ) -> None: ...

    def set_zbound(
        self,
        lower: float | None,
        upper: float | None,
        view_margin: float | None
    ) -> None: ...

    def _set_lim3d(
        self,
        axis,
        lower, upper,
        emit, auto, view_margin,
        axmin, axmax): ...

    def set_xlim(
        self,
        left: float | None,
        right: float | None,
        emit: bool,
        auto: bool | None,
        view_margin: float | None,
        xmin: float | None,
        xmax: float | None
    ) -> tuple[float, float]: ...

    def set_ylim(
        self,
        left: float | None,
        right: float | None,
        emit: bool,
        auto: bool | None,
        view_margin: float | None,
        xmin: float | None,
        xmax: float | None
    ) -> tuple[float, float]: ...

    def set_zlim(
        self,
        left: float | None,
        right: float | None,
        emit: bool,
        auto: bool | None,
        view_margin: float | None,
        xmin: float | None,
        xmax: float | None
    ) -> tuple[float, float]: ...

    def get_xlim(self) -> tuple: ...

    def get_ylim(self) -> tuple: ...

    def get_zlim(self) -> tuple: ...

    def view_init(
            self,
            elev: float | None,
            azim: float | None,
            roll: float | None,
            vertical_axis: str,
            share: bool
        ) -> None: ...

    def set_proj_type(
            self,
            proj_type: str,
            focal_length: float | None
        ) -> None: ...

    def _roll_to_vertical(
            self,
            arr: ArrayLike,
            reverse: bool
    ) -> np.ndarray: ...

    def get_proj(self) -> np.ndarray: ...

    def mouse_init(
            self,
            rotate_btn: int | list,
            pan_btn: int | list,
            zoom_btn: int | list
        ) -> None: ...

    def disable_mouse_rotation(self) -> None: ...

    def can_zoom(self) -> bool: ...

    def can_pan(self) -> bool: ...

    def sharez(self, other: "Axes3D") -> None: ...

    def shareview(self, other: "Axes3D") -> None: ...

    def clear(self) -> None: ...

    def format_zdata(self, z) -> str: ...

    def format_coord(
            self,
            xv: float,
            yv: float,
            renderer
        ) -> str: ...

    def _rotation_coords(self) -> str: ...

    def _location_coords(self, xv, yv, renderer): ...

    def _get_camera_loc(self) -> ArrayLike: ...

    def _calc_coord(
        self,
        xv: float | ArrayLike,
        yv: float | ArrayLike,
        renderer: None
    ) -> tuple[ArrayLike, np.intp]: ...

    def _arcball(
        self,
        x: float,
        y: float
    ) -> np.ndarray: ...

    def _on_move(self, event) -> None: ...

    def drag_pan(
        self,
        button: MouseButton,
        key: str,
        x: float,
        y: float
    ) -> None: ...

    def _calc_view_axes(
        self,
        eye: np.ndarray
    ) -> tuple: ...

    def _set_view_from_bbox(
        self,
        bbox,
        direction: str,
        mode: str | None,
        twinx: bool,
        twiny: bool
    ) -> None: ...

    def _zoom_data_limits(
        self,
        scale_u: float,
        scale_y: float,
        scale_w: float
    ) -> None: ...

    def _scale_axis_limits(
        self,
        scale_x: float,
        scale_y: float,
        scale_z: float
    ) -> None: ...

    def _get_w_centers_ranges(self) -> tuple[
        float, float, float, float, float, float]: ...

    def set_zlabel(
        self,
        zlabel: str,
        fontdict: dict | None,
        labelpad: float | None
    ) -> Text: ...

    def get_zlabel(self) -> str: ...

    def grid(self, visible: bool) -> None: ...

    def tick_params(self, axis: str) -> None: ...

    def invert_zaxis(self) -> None: ...

    def get_zbound(self) -> Tuple[float, float]: ...

    def text(
        self,
        x: float, y: float, z: float,
        s: str,
        zdir: None | Literal['x'] | Literal['y'] | Literal['z'] | tuple,
        axlim_clip: bool
    ) -> Text: ...

    def plot(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
        zdir: str,
        axlim_clip: bool,
    ) -> list[Artist]:  ...

    def fill_between(
        self,
        x1: float | ArrayLike,
        y1: float | ArrayLike,
        z1: float | ArrayLike,
        x2: float | ArrayLike,
        y2: float | ArrayLike,
        z2: float | ArrayLike,
        where: list[bool],
        mode: str,
        axlim_clip: bool
    ) -> Poly3DCollection: ...

    def plot_surface(
        self,
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike,
        norm: Normalize | str | None,
        vmin: float | None,
        vmax: float | None,
        lightsource: None | LightSource,
        axlim_clip: bool
    ) -> Poly3DCollection: ...

    def plot_wireframe(
        self,
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike,
        axlim_clip: bool
    ) -> Line3DCollection: ...

    def plot_trisurf(
        self,
        color: None | np.ndarray,
        norm: Normalize,
        vmin: float | None, vmax: float | None,
        lightsource: None | LightSource,
        axlim_clip: bool
    ) -> Poly3DCollection: ...

    def _3d_extend_contour(
        self,
        cset: QuadContourSet,
        stride: int
    ) -> None: ...

    def add_contour_set(
        self,
        cset: QuadContourSet,
        extend3d: bool,
        stride: int,
        zdir: str,
        offset: int | None,
        axlim_clip: None | bool
    ) -> None: ...

    def add_contourf_set(
        self,
        cset: QuadContourSet,
        zdir: str,
        offset: int | None,
        axlim_clip: bool
    ) -> None: ...

    def _add_contourf_set(
        self,
        cset: QuadContourSet,
        zdir: str,
        offset: int | None,
        axlim_clip: bool
    ) -> np.ndarray: ...

    def contour(
        self,
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike,
        extend3d: bool,
        stride: int,
        zdir: str,
        offset: int | None,
        axlim_clip: bool
    ) -> QuadContourSet: ...

    def tricontour(
        self,
        extend3d: bool,
        stride: int,
        zdir: str,
        offset: None | float,
        axlim_clip: bool
    ) -> TriContourSet: ...

    def contourf(
        self,
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike,
        zdir: str,
        offset: int | None,
        axlim_clip: bool
    ) -> QuadContourSet: ...

    def tricontourf(
        self,
        zdir: str,
        offset: None | float,
        axlim_clip: bool
    ) -> TriContourSet: ...

    def add_collection3d(
        self,
        col: Collection,
        zs: float | ArrayLike,
        zdir: str,
        autolim: bool,
        axlim_clip: bool
    ) -> Collection: ...
        
    
    def scatter(
        self,
        xs: ArrayLike, ys: ArrayLike, 
        zs: float | ArrayLike, 
        zdir: str,
        s: float | ArrayLike,
        c: ArrayLike | Sequence[ColorType] | ColorType | None,
        depthshade: bool,
        depthshade_minalpha: float,
        axlim_clip: bool
    ) -> PathCollection: ...

    def _auto_scale_contourf(
        self, 
        X, Y, Z,
        zdir: str,
        levels,
        had_data: bool
    ) -> None: ...



    def get_tightbbox(
            self,
            renderer,
            call_axes_locator: bool,
            bbox_extra_artists,
            for_layout_only: bool
    ) -> Bbox: ...


def get_test_data(
    delta: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


class _Quaternion:
    scalar: float
    vector: np.ndarray

    def __init__(self, scalar: float, vector: ArrayLike) -> None: ...

    def __neg__(self) -> "_Quaternion": ...

    def __mul__(self, other: "_Quaternion") -> "_Quaternion": ...

    def conjugate(self) -> "_Quaternion": ...

    @property
    def norm(self) -> float: ...

    def normalize(self) -> "_Quaternion": ...

    def reciprocal(self) -> "_Quaternion": ...

    def __truediv__(self, other: "_Quaternion") -> "_Quaternion": ...

    def rotate(self, v: ArrayLike) -> np.ndarray: ...

    def __eq__(self, other: object) -> bool: ...

    def __repr__(self) -> str: ...

    @classmethod
    def rotate_from_to(cls, r1: ArrayLike, r2: ArrayLike) -> "_Quaternion": ...

    @classmethod
    def from_cardan_angles(
        cls,
        elev: float,
        azim: float,
        roll: float
    ) -> "_Quaternion":
        ...

    def as_cardan_angles(self) -> Tuple[float, float, float]: ...
