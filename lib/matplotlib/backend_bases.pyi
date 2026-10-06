from contextlib import AbstractContextManager
from enum import Enum, IntEnum
import os
import socket
from matplotlib import (
    cbook,
    transforms,
    widgets,
    _api,
)
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_managers import ToolManager
from matplotlib.backend_tools import Cursors, ToolBase
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.text import Text, TextToPath
from matplotlib.transforms import Bbox, BboxBase, Transform, TransformedPath

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, IO, Literal, NamedTuple, Self, overload
from typing_extensions import TypeVar
from numpy.typing import ArrayLike
from .typing import (
    BlendModeType,
    CapStyleType,
    CloseEventType,
    ColorType,
    DrawEventType,
    FillRuleType,
    JoinStyleType,
    KeyEventType,
    LineStyleType,
    MouseEventType,
    PickEventType,
    ResizeEventType,
)

# Type parameters for narrowing cross-references between backend classes.
#
# A backend is a cooperating set of classes that reference each other:
#
#   FigureCanvasBase[ToolbarT, ManagerT, TimerT]
#       .toolbar -> ToolbarT | None
#       .manager -> ManagerT | None
#       .new_timer() -> TimerT
#   NavigationToolbar2[CanvasT]
#       .canvas -> CanvasT
#   FigureManagerBase[CanvasT, ToolbarT]
#       .canvas -> CanvasT
#       .toolbar -> ToolbarT | ToolContainerBase | None
#
# A backend declares its concrete set once, e.g.::
#
#   class FigureCanvasFoo(FigureCanvasBase["NavigationToolbar2Foo",
#                                          "FigureManagerFoo", TimerFoo]): ...
#   class NavigationToolbar2Foo(NavigationToolbar2[FigureCanvasFoo]): ...
#   class FigureManagerFoo(FigureManagerBase[FigureCanvasFoo,
#                                            NavigationToolbar2Foo]): ...
#
# and every cross-reference then resolves to the concrete type without
# redeclaring the (invariant, mutable) attributes in the subclass.
#
# CanvasT intentionally has no default: its recursive invariant bound would
# overconstrain concrete toolbar and manager subclasses.  The remaining type
# variables default to ``Any`` specializations of their bounds so that bare
# ``FigureCanvasBase`` / ``FigureManagerBase`` remain usable and accept any
# concrete backend.  General-purpose consumers (``Figure.canvas``, events,
# handlers) should spell this out as ``FigureCanvasBase[Any, Any, Any]``.
# ``None`` is in the annotation, not the TypeVar.
CanvasT = TypeVar("CanvasT", bound="FigureCanvasBase[Any, Any, Any]")

class _UntypedToolkitWidget:
    """
    Stand-in for an untyped native GUI base class (Qt, GTK, wx).

    GUI backend classes inherit their toolkit widget at runtime
    (``QtWidgets.QWidget``, ``Gtk.DrawingArea``, ``wx.Panel``, ...).  Those
    toolkits ship no type information, so the backend stubs list this class
    in place of the native base.  Any attribute not declared by Matplotlib
    resolves to ``Any``, exactly as it would when type checking against the
    runtime module, while Matplotlib-owned members stay precisely typed.

    This is a typing-only construct and does not exist at runtime.
    """

    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
ToolbarT = TypeVar(
    "ToolbarT", bound="NavigationToolbar2[Any]", default="NavigationToolbar2[Any]"
)
ManagerT = TypeVar(
    "ManagerT",
    bound="FigureManagerBase[Any, Any]",
    default="FigureManagerBase[Any, Any]",
)
# CanvasClassT is the concrete FigureCanvas *class* (not instance) stored on
# a _Backend subclass.  default lets bare _Backend remain valid.
CanvasClassT = TypeVar(
    "CanvasClassT",
    bound="type[FigureCanvasBase[Any, Any, Any]]",
    default="type[FigureCanvasBase[Any, Any, Any]]",
)
# GCT lets RendererBase subclasses declare which GraphicsContext subclass they
# produce/consume.  Covariant usage (new_gc return type) is always safe;
# contravariant usage (gc: GCT parameters) is safe within a subclass because
# the renderer always pairs with its own GC via self.new_gc().
GCT = TypeVar("GCT", bound="GraphicsContextBase", default="GraphicsContextBase")
# TimerT lets FigureCanvasBase subclasses declare the concrete TimerBase
# subclass returned by new_timer().
TimerT = TypeVar("TimerT", bound="TimerBase", default="TimerBase")

def register_backend(
    format: str,
    backend: str | type[FigureCanvasBase[Any, Any, Any]],
    description: str | None = ...,
) -> None: ...
def get_registered_canvas_class(
    format: str,
) -> type[FigureCanvasBase[Any, Any, Any]]: ...

class RendererBase(Generic[GCT]):
    _text2path: TextToPath
    def __init__(self) -> None: ...
    def open_group(self, s: str, gid: str | None = ...) -> None: ...
    def close_group(self, s: str) -> None: ...
    def open_blend_group(
        self,
        blend_mode: BlendModeType | None,
        *,
        alpha: float = ...,
        knockout: bool = ...,
    ) -> None: ...
    def close_blend_group(self) -> None: ...
    def draw_path(
        self,
        gc: GCT,
        path: Path,
        transform: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GCT,
        marker_path: Path,
        marker_trans: Transform,
        path: Path,
        trans: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GCT,
        master_transform: Transform,
        paths: Sequence[Path],
        all_transforms: Sequence[ArrayLike],
        offsets: ArrayLike | Sequence[ArrayLike],
        offset_trans: Transform,
        facecolors: ColorType | Sequence[ColorType],
        edgecolors: ColorType | Sequence[ColorType],
        linewidths: float | Sequence[float],
        linestyles: LineStyleType | Sequence[LineStyleType],
        antialiaseds: bool | Sequence[bool],
        urls: str | Sequence[str],
        offset_position: Any,
        *,
        hatchcolors: ColorType | Sequence[ColorType] | None = None,
    ) -> None: ...
    def draw_quad_mesh(
        self,
        gc: GCT,
        master_transform: Transform,
        meshWidth: int,
        meshHeight: int,
        coordinates: ArrayLike,
        offsets: ArrayLike | Sequence[ArrayLike],
        offsetTrans: Transform,
        facecolors: Sequence[ColorType],
        antialiased: bool,
        edgecolors: Sequence[ColorType] | ColorType | None,
    ) -> None: ...
    def draw_gouraud_triangles(
        self,
        gc: GCT,
        triangles_array: ArrayLike,
        colors_array: ArrayLike,
        transform: Transform,
    ) -> None: ...
    def get_image_magnification(self) -> float: ...
    def draw_image(
        self,
        gc: GCT,
        x: float,
        y: float,
        im: ArrayLike,
        transform: transforms.Affine2DBase | None = ...,
    ) -> None: ...
    def option_image_nocomposite(self) -> bool: ...
    def option_scale_image(self) -> bool: ...
    def draw_tex(
        self,
        gc: GCT,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        *,
        mtext: Text | None = ...,
    ) -> None: ...
    def draw_text(
        self,
        gc: GCT,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        ismath: bool | Literal["TeX"] = ...,
        mtext: Text | None = ...,
    ) -> None: ...
    def get_text_width_height_descent(
        self, s: str, prop: FontProperties, ismath: bool | Literal["TeX"]
    ) -> tuple[float, float, float]: ...
    def flipy(self) -> bool: ...
    def get_canvas_width_height(self) -> tuple[float, float]: ...
    def get_texmanager(self) -> TexManager: ...
    def new_gc(self) -> GCT: ...
    def points_to_pixels(self, points: ArrayLike) -> ArrayLike: ...
    def start_rasterizing(self) -> None: ...
    def stop_rasterizing(self) -> None: ...
    def start_filter(self) -> None: ...
    def stop_filter(
        self,
        filter_func: Callable[[ArrayLike, float], tuple[ArrayLike, float, float]],
    ) -> None: ...

class GraphicsContextBase:
    def __init__(self) -> None: ...
    def copy_properties(self, gc: GraphicsContextBase) -> None: ...
    def restore(self) -> None: ...
    def get_alpha(self) -> float: ...
    def get_blend_mode(self) -> str: ...
    def get_fill_rule(self) -> FillRuleType: ...
    def get_antialiased(self) -> int: ...
    def get_capstyle(self) -> Literal["butt", "projecting", "round"]: ...
    def get_clip_rectangle(self) -> Bbox | None: ...
    def get_clip_path(
        self,
    ) -> tuple[TransformedPath, Transform] | tuple[None, None]: ...
    def get_dashes(self) -> tuple[float, ArrayLike | None]: ...
    def get_forced_alpha(self) -> bool: ...
    def get_joinstyle(self) -> Literal["miter", "round", "bevel"]: ...
    def get_linewidth(self) -> float: ...
    def get_rgb(self) -> tuple[float, float, float, float]: ...
    def get_url(self) -> str | None: ...
    def get_gid(self) -> int | None: ...
    def get_snap(self) -> bool | None: ...
    def set_alpha(self, alpha: float) -> None: ...
    def set_blend_mode(self, blend_mode: BlendModeType) -> None: ...
    def set_fill_rule(self, fill_rule: FillRuleType) -> None: ...
    def set_antialiased(self, b: bool) -> None: ...
    def set_capstyle(self, cs: CapStyleType) -> None: ...
    def set_clip_rectangle(self, rectangle: Bbox | None) -> None: ...
    def set_clip_path(self, path: TransformedPath | None) -> None: ...
    def set_dashes(self, dash_offset: float, dash_list: ArrayLike | None) -> None: ...
    def set_foreground(self, fg: ColorType, isRGBA: bool = ...) -> None: ...
    def set_joinstyle(self, js: JoinStyleType) -> None: ...
    def set_linewidth(self, w: float) -> None: ...
    def set_url(self, url: str | None) -> None: ...
    def set_gid(self, id: int | None) -> None: ...
    def set_snap(self, snap: bool | None) -> None: ...
    def set_hatch(self, hatch: str | None) -> None: ...
    def get_hatch(self) -> str | None: ...
    def get_hatch_path(self, density: float = ...) -> Path: ...
    def get_hatch_color(self) -> ColorType: ...
    def set_hatch_color(self, hatch_color: ColorType) -> None: ...
    def get_hatch_linewidth(self) -> float: ...
    def set_hatch_linewidth(self, hatch_linewidth: float) -> None: ...
    def get_sketch_params(self) -> tuple[float, float, float] | None: ...
    def set_sketch_params(
        self,
        scale: float | None = ...,
        length: float | None = ...,
        randomness: float | None = ...,
    ) -> None: ...

class TimerBase:
    callbacks: list[tuple[Callable, tuple, dict[str, Any]]]
    def __init__(
        self,
        interval: int | None = ...,
        callbacks: list[tuple[Callable, tuple, dict[str, Any]]] | None = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def interval(self) -> int: ...
    @interval.setter
    def interval(self, interval: int) -> None: ...
    @property
    def single_shot(self) -> bool: ...
    @single_shot.setter
    def single_shot(self, ss: bool) -> None: ...
    def add_callback(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any]: ...
    def remove_callback(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None: ...

class Event:
    name: str
    canvas: FigureCanvasBase[Any, Any, Any]
    guiEvent: Any
    def __init__(
        self, name: str, canvas: FigureCanvasBase[Any, Any, Any], guiEvent: Any | None = ...
    ) -> None: ...
    # Backend-authoring API: dispatch this event on ``self.canvas``.
    def _process(self) -> None: ...

class DrawEvent(Event):
    renderer: RendererBase[Any]
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase[Any, Any, Any],
        renderer: RendererBase[Any],
    ) -> None: ...

class ResizeEvent(Event):
    width: int
    height: int
    def __init__(
        self, name: str, canvas: FigureCanvasBase[Any, Any, Any]
    ) -> None: ...

class CloseEvent(Event): ...

class LocationEvent(Event):
    x: int
    y: int
    inaxes: Axes | None
    xdata: float | None
    ydata: float | None
    modifiers: frozenset[str]
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase[Any, Any, Any],
        x: int,
        y: int,
        guiEvent: Any | None = ...,
        *,
        modifiers: Iterable[str] | None = ...,
    ) -> None: ...

class MouseButton(IntEnum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    BACK = 8
    FORWARD = 9

class MouseEvent(LocationEvent):
    button: MouseButton | Literal["up", "down"] | None
    key: str | None
    step: float
    dblclick: bool
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase[Any, Any, Any],
        x: int,
        y: int,
        button: MouseButton | Literal["up", "down"] | None = ...,
        key: str | None = ...,
        step: float = ...,
        dblclick: bool = ...,
        guiEvent: Any | None = ...,
        *,
        buttons: Iterable[MouseButton] | None = ...,
        modifiers: Iterable[str] | None = ...,
    ) -> None: ...

class PickEvent(Event):
    mouseevent: MouseEvent
    artist: Artist
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase[Any, Any, Any],
        mouseevent: MouseEvent,
        artist: Artist,
        guiEvent: Any | None = ...,
        **kwargs: Any,
    ) -> None: ...

class KeyEvent(LocationEvent):
    key: str | None
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase[Any, Any, Any],
        key: str | None,
        x: int = ...,
        y: int = ...,
        guiEvent: Any | None = ...,
    ) -> None: ...

class FigureCanvasBase(Generic[ToolbarT, ManagerT, TimerT]):
    required_interactive_framework: str | None

    @_api.classproperty
    def manager_class(cls) -> type[ManagerT]: ...
    events: list[str]
    fixed_dpi: None | float
    filetypes: dict[str, str]

    @_api.classproperty
    def supports_blit(cls) -> bool: ...

    figure: Figure
    manager: ManagerT | None
    # Backend-authoring API: the timer class instantiated by ``new_timer``.
    _timer_cls: type[TimerT]
    widgetlock: widgets.LockDraw
    mouse_grabber: None | Axes
    toolbar: ToolbarT | None
    def __init__(self, figure: Figure | None = ...) -> None: ...
    @property
    def callbacks(self) -> cbook.CallbackRegistry: ...
    @property
    def button_pick_id(self) -> int: ...
    @property
    def scroll_pick_id(self) -> int: ...
    @classmethod
    def new_manager(cls, figure: Figure, num: int | str) -> ManagerT: ...
    def is_saving(self) -> bool: ...
    def blit(self, bbox: BboxBase | None = ...) -> None: ...
    def inaxes(self, xy: tuple[float, float]) -> Axes | None: ...
    def grab_mouse(self, ax: Axes) -> None: ...
    def release_mouse(self, ax: Axes) -> None: ...
    def set_cursor(self, cursor: Cursors) -> None: ...
    # Runtime accepts and ignores ``*args, **kwargs``; every backend implements
    # ``(self)``, which is the form declared here so that subclasses match.
    def draw(self) -> None: ...
    def draw_idle(self) -> None: ...
    @property
    def device_pixel_ratio(self) -> float: ...
    # Backend-authoring API: record the physical/logical pixel ratio.
    def _set_device_pixel_ratio(self, ratio: float) -> bool: ...
    def get_width_height(self, *, physical: bool = ...) -> tuple[int, int]: ...
    @classmethod
    def get_supported_filetypes(cls) -> dict[str, str]: ...
    @classmethod
    def get_supported_filetypes_grouped(cls) -> dict[str, list[str]]: ...
    def print_figure(
        self,
        filename: str | os.PathLike | IO,
        dpi: float | None = ...,
        facecolor: ColorType | Literal["auto"] | None = ...,
        edgecolor: ColorType | Literal["auto"] | None = ...,
        orientation: str = ...,
        format: str | None = ...,
        *,
        bbox_inches: Literal["tight"] | Bbox | None = ...,
        pad_inches: float | None = ...,
        bbox_extra_artists: list[Artist] | None = ...,
        backend: str | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def get_default_filename(self) -> str: ...
    @overload
    def mpl_connect(
        self,
        s: MouseEventType,
        func: Callable[[MouseEvent], Any],
    ) -> int: ...
    @overload
    def mpl_connect(
        self,
        s: KeyEventType,
        func: Callable[[KeyEvent], Any],
    ) -> int: ...
    @overload
    def mpl_connect(
        self, s: PickEventType, func: Callable[[PickEvent], Any]
    ) -> int: ...
    @overload
    def mpl_connect(
        self, s: ResizeEventType, func: Callable[[ResizeEvent], Any]
    ) -> int: ...
    @overload
    def mpl_connect(
        self, s: CloseEventType, func: Callable[[CloseEvent], Any]
    ) -> int: ...
    @overload
    def mpl_connect(
        self, s: DrawEventType, func: Callable[[DrawEvent], Any]
    ) -> int: ...
    def mpl_disconnect(self, cid: int) -> None: ...
    def new_timer(
        self,
        interval: int | None = ...,
        callbacks: list[tuple[Callable, tuple, dict[str, Any]]] | None = ...,
    ) -> TimerT: ...
    def flush_events(self) -> None: ...
    def start_event_loop(self, timeout: float = ...) -> None: ...
    def stop_event_loop(self) -> None: ...

def key_press_handler(
    event: KeyEvent,
    canvas: FigureCanvasBase[Any, Any, Any] | None = ...,
    toolbar: NavigationToolbar2[Any] | None = ...,
) -> None: ...
def button_press_handler(
    event: MouseEvent,
    canvas: FigureCanvasBase[Any, Any, Any] | None = ...,
    toolbar: NavigationToolbar2[Any] | None = ...,
) -> None: ...
def scroll_handler(
    event: MouseEvent,
    canvas: FigureCanvasBase[Any, Any, Any] | None = ...,
    toolbar: NavigationToolbar2[Any] | None = ...,
) -> None: ...

class NonGuiException(Exception): ...

class FigureManagerBase(Generic[CanvasT, ToolbarT]):
    canvas: CanvasT
    num: int | str
    key_press_handler_id: int | None
    button_press_handler_id: int | None
    scroll_handler_id: int | None
    toolmanager: ToolManager | None
    toolbar: ToolbarT | ToolContainerBase | None
    _toolbar2_class: type[ToolbarT] | None
    _toolmanager_toolbar_class: type[ToolContainerBase] | None
    def __init__(self, canvas: CanvasT, num: int | str) -> None: ...
    @classmethod
    def create_with_canvas(
        cls, canvas_class: type[CanvasT], figure: Figure, num: int | str
    ) -> Self: ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    @classmethod
    def pyplot_show(cls, *, block: bool | None = ...) -> None: ...
    def show(self) -> None: ...
    def destroy(self) -> None: ...
    def full_screen_toggle(self) -> None: ...
    def resize(self, w: int, h: int) -> None: ...
    def get_window_title(self) -> str: ...
    def set_window_title(self, title: str) -> None: ...

cursors = Cursors

class _Mode(str, Enum):
    NONE = ""
    PAN = "pan/zoom"
    ZOOM = "zoom rect"

class NavigationToolbar2(Generic[CanvasT]):
    toolitems: Sequence[tuple[str | None, ...]]
    UNKNOWN_SAVED_STATUS: object
    canvas: CanvasT
    mode: _Mode
    def __init__(self, canvas: CanvasT) -> None: ...
    def set_message(self, s: str) -> None: ...
    def draw_rubberband(
        self, event: Event, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...
    def remove_rubberband(self) -> None: ...
    def home(self, *args: Any) -> None: ...
    def back(self, *args: Any) -> None: ...
    def forward(self, *args: Any) -> None: ...
    def mouse_move(self, event: MouseEvent) -> None: ...
    def pan(self, *args: Any) -> None: ...

    class _PanInfo(NamedTuple):
        button: MouseButton
        axes: list[Axes]
        cid: int

    def press_pan(self, event: Event) -> None: ...
    def drag_pan(self, event: Event) -> None: ...
    def release_pan(self, event: Event) -> None: ...
    def zoom(self, *args: Any) -> None: ...

    class _ZoomInfo(NamedTuple):
        button: MouseButton
        start_xy: tuple[float, float]
        axes: list[Axes]
        cid: int
        cbar: Colorbar

    def press_zoom(self, event: Event) -> None: ...
    def drag_zoom(self, event: Event) -> None: ...
    def release_zoom(self, event: Event) -> None: ...
    def push_current(self) -> None: ...
    subplot_tool: widgets.SubplotTool
    def configure_subplots(self, *args: Any) -> widgets.SubplotTool: ...
    def save_figure(self, *args: Any) -> str | None | object: ...
    def update(self) -> None: ...
    def set_history_buttons(self) -> None: ...

class ToolContainerBase:
    toolmanager: ToolManager
    def __init__(self, toolmanager: ToolManager) -> None: ...
    def add_tool(self, tool: ToolBase, group: str, position: int = ...) -> None: ...
    def trigger_tool(self, name: str) -> None: ...
    def add_toolitem(
        self,
        name: str,
        group: str,
        position: int,
        image: str,
        description: str,
        toggle: bool,
    ) -> None: ...
    def toggle_toolitem(self, name: str, toggled: bool) -> None: ...
    def remove_toolitem(self, name: str) -> None: ...
    def set_message(self, s: str) -> None: ...

class _Backend(Generic[CanvasClassT]):
    backend_version: str | float
    FigureCanvas: CanvasClassT | None
    FigureManager: type[FigureManagerBase[Any, Any]]
    mainloop: None | Callable[[], Any]
    @classmethod
    def new_figure_manager(
        cls, num: int | str, *args: Any, **kwargs: Any
    ) -> FigureManagerBase[Any, Any]: ...
    @classmethod
    def new_figure_manager_given_figure(
        cls, num: int | str, figure: Figure
    ) -> FigureManagerBase[Any, Any]: ...
    @classmethod
    def draw_if_interactive(cls) -> None: ...
    @classmethod
    def show(cls, *, block: bool | None = ...) -> None: ...
    # ``export`` is a ``staticmethod`` used as a class decorator; its single
    # parameter is the class being decorated.
    @staticmethod
    def export(cls: type[_Backend[Any]]) -> type[_Backend[Any]]: ...

class ShowBase(_Backend[type[FigureCanvasBase[Any, Any, Any]]]):
    def __call__(self, block: bool | None = ...) -> None: ...

def _allow_interrupt(
    prepare_notifier: Callable[[socket.socket], object],
    handle_sigint: Callable[[object], object],
) -> AbstractContextManager[None]: ...
