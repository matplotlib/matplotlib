"""
Static and runtime checks for the backend typing model.

The ``_check_*`` helpers in this module are analyzed by mypy (this file is
listed explicitly in ``[tool.mypy] files``) but never executed by pytest.
They exercise the generic cross-references between ``FigureCanvasBase``,
``NavigationToolbar2``, ``FigureManagerBase``, and ``RendererBase`` from the
point of view of

* ordinary downstream code that passes concrete backend objects through the
  general-purpose Matplotlib API, and
* a third-party backend author who subclasses the base classes (including the
  "mixin + toolkit" diamond pattern).

Negative checks use ``# type: ignore[<code>]`` together with the
``warn_unused_ignores`` override for this module in ``pyproject.toml``: if a
declaration ever stops rejecting an invalid call, the now-unused ignore is
reported as an error.
"""

from __future__ import annotations

import datetime
import io
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_type

import matplotlib.backend_bases as backend_bases
from matplotlib.artist import Artist
from matplotlib.backend_bases import (
    CloseEvent,
    DrawEvent,
    Event,
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    KeyEvent,
    LocationEvent,
    MouseEvent,
    NavigationToolbar2,
    PickEvent,
    ResizeEvent,
    RendererBase,
    TimerBase,
    ToolContainerBase,
    _Backend,
    button_press_handler,
    get_registered_canvas_class,
    key_press_handler,
    register_backend,
    scroll_handler,
)
from matplotlib.backend_managers import ToolManager
from matplotlib.backend_tools import (
    ToolBase,
    _find_tool_class,
    _register_tool_class,
    _tool_registry,
)
from matplotlib.backends._backend_agg import BufferRegion
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
    FigureCanvasPdf,
    FigureManagerPdf,
    Name,
    PdfFile,
    PdfPages,
    RendererPdf,
)
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.backends.backend_template import (
    FigureCanvasTemplate,
    FigureManagerTemplate,
)
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage, composite_images
from matplotlib.inset import InsetIndicator
from matplotlib.lines import VertexSelector
from matplotlib.offsetbox import DraggableBase, OffsetBox
from matplotlib.patches import bbox_artist, draw_bbox
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import PathEffectRenderer
from matplotlib.table import Cell
from matplotlib.texmanager import TexManager
from matplotlib.text import Annotation, OffsetFrom
from matplotlib.transforms import Bbox, Transform
from matplotlib.widgets import AxesWidget

if TYPE_CHECKING:
    # GUI backends need optional native dependencies; only their stubs are
    # exercised here, so import them for the type checker alone.
    from matplotlib.backends._backend_tk import (
        FigureCanvasTk,
        FigureManagerTk,
        NavigationToolbar2Tk,
        TimerTk,
    )
    from matplotlib.backends._backend_gtk import TimerGTK
    from matplotlib.backends.backend_cairo import FigureCanvasCairo, RendererCairo
    from matplotlib.backends.backend_gtk3 import (
        FigureCanvasGTK3,
        FigureManagerGTK3,
        NavigationToolbar2GTK3,
    )
    from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg
    from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo
    from matplotlib.backends.backend_gtk4 import (
        FigureCanvasGTK4,
        FigureManagerGTK4,
        NavigationToolbar2GTK4,
    )
    from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg
    from matplotlib.backends.backend_gtk4cairo import FigureCanvasGTK4Cairo
    from matplotlib.backends.backend_nbagg import (
        FigureCanvasNbAgg,
        FigureManagerNbAgg,
        NavigationIPy,
    )
    from matplotlib.backends.backend_qt import (
        FigureCanvasQT,
        FigureManagerQT,
        NavigationToolbar2QT,
        TimerQT,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qtcairo import FigureCanvasQTCairo
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.backends.backend_tkcairo import FigureCanvasTkCairo
    from matplotlib.backends.backend_webagg import (
        FigureCanvasWebAgg,
        FigureManagerWebAgg as FigureManagerWebAggApplication,
    )
    from matplotlib.backends.backend_webagg_core import (
        FigureCanvasWebAggCore,
        FigureManagerWebAgg as FigureManagerWebAggCore,
        NavigationToolbar2WebAgg,
        TimerAsyncio,
    )
    from matplotlib.backends.backend_wx import (
        FigureManagerWx,
        NavigationToolbar2Wx,
        TimerWx,
        _FigureCanvasWxBase,
    )
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
    from matplotlib.backends.backend_wxcairo import FigureCanvasWxCairo


# ---------------------------------------------------------------------------
# A minimal in-tree "specialized" Agg backend used by the runtime tests.
# ---------------------------------------------------------------------------


class SpecializedTimer(TimerBase):
    pass


class SpecializedFigureCanvasAgg(
    FigureCanvasAgg[
        NavigationToolbar2[Any], FigureManagerBase[Any, Any], SpecializedTimer
    ]
):
    pass


class SpecializedTool(ToolBase):
    pass


class SpecializedFigureManager(FigureManagerBase[Any, Any]):
    pass


# ---------------------------------------------------------------------------
# A simulated third-party backend: the pattern from a downstream project that
# subclasses the four cooperating base classes and then mixes the result into
# the stock Qt and Tk backends.  Defined at module level so that mypy checks
# the class statements themselves (invariant attribute overrides, diamond
# base compatibility), not just expressions.
# ---------------------------------------------------------------------------


class RemoteTimer(TimerBase):
    pass


class RemoteGraphicsContext(GraphicsContextBase):
    pass


class RemoteRenderer(RendererBase[RemoteGraphicsContext]):
    def new_gc(self) -> RemoteGraphicsContext:
        return RemoteGraphicsContext()

    def draw_path(
        self,
        gc: RemoteGraphicsContext,
        path: MplPath,
        transform: Transform,
        rgbFace: Any = None,
    ) -> None:
        pass


class FigureCanvasRemote(
    FigureCanvasBase["NavigationToolbar2Remote", "FigureManagerRemote", RemoteTimer]
):
    _timer_cls = RemoteTimer

    def _use_backend_authoring_api(self) -> None:  # pragma: no cover
        assert_type(self._set_device_pixel_ratio(2.0), bool)
        Event("remote_event", self)._process()
        MouseEvent("button_press_event", self, 1, 2)._process()


class NavigationToolbar2Remote(NavigationToolbar2[FigureCanvasRemote]):
    pass


class FigureManagerRemote(
    FigureManagerBase[FigureCanvasRemote, NavigationToolbar2Remote]
):
    _toolbar2_class = NavigationToolbar2Remote


class _BackendRemote(_Backend[type[FigureCanvasRemote]]):
    FigureCanvas = FigureCanvasRemote
    FigureManager = FigureManagerRemote


if TYPE_CHECKING:
    # Diamond: remote mixin first, stock toolkit class second, for both Qt
    # and Tk (whose toolbar has an incompatible ``message`` attribute type at
    # runtime).  A thin client has no local renderer, so it pairs with the
    # bare toolkit canvas (``FigureCanvasQT``, not ``FigureCanvasQTAgg``) in
    # the same way the in-tree ``FigureCanvasAgg`` does.  mypy accepts these;
    # pyright reports the two parameterizations of the shared generic base as
    # "mutually incompatible" by design, so the class statements carry a
    # pyright-only suppression.

    class FigureCanvasQTRemote(FigureCanvasRemote, FigureCanvasQT):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    class NavigationToolbar2QTRemote(NavigationToolbar2Remote, NavigationToolbar2QT):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    class FigureManagerQTRemote(FigureManagerRemote, FigureManagerQT):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    class FigureCanvasTkRemote(FigureCanvasRemote, FigureCanvasTk):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    class NavigationToolbar2TkRemote(NavigationToolbar2Remote, NavigationToolbar2Tk):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    class FigureManagerTkRemote(FigureManagerRemote, FigureManagerTk):  # pyright: ignore[reportGeneralTypeIssues]
        pass

    # Narrowing a stock toolkit backend further.
    class MyQtCanvas(FigureCanvasQTAgg):
        def _check(self) -> None:  # pragma: no cover
            assert_type(self.manager, FigureManagerQT | None)
            assert_type(self.toolbar, NavigationToolbar2QT | None)
            assert_type(self.new_timer(), TimerQT)
            assert_type(type(self).manager_class, type[FigureManagerQT])
            region = self.copy_from_bbox(self.figure.bbox)
            assert_type(region, BufferRegion)
            self.restore_region(region)
            self.renderer.clear()
            assert_type(self._lastKey, tuple[float, float, float] | None)


# ---------------------------------------------------------------------------
# Static checks (analyzed by mypy, never called).
# ---------------------------------------------------------------------------


def _check_remote_backend_types(  # pragma: no cover
    canvas: FigureCanvasRemote,
    toolbar: NavigationToolbar2Remote,
    manager: FigureManagerRemote,
    figure: Figure,
) -> None:
    assert_type(canvas.toolbar, NavigationToolbar2Remote | None)
    assert_type(canvas.manager, FigureManagerRemote | None)
    assert_type(canvas.new_timer(), RemoteTimer)
    assert_type(FigureCanvasRemote.manager_class, type[FigureManagerRemote])
    assert_type(FigureCanvasRemote.new_manager(figure, 1), FigureManagerRemote)

    assert_type(toolbar.canvas, FigureCanvasRemote)
    assert_type(toolbar.canvas.manager, FigureManagerRemote | None)

    assert_type(manager.canvas, FigureCanvasRemote)
    assert_type(
        manager.toolbar, NavigationToolbar2Remote | ToolContainerBase | None
    )
    # Assigned (not annotated) in the class body; mypy infers the narrowed
    # declared type from the assignment, pyright keeps the inherited one.
    assert manager._toolbar2_class is NavigationToolbar2Remote
    assert_type(
        FigureManagerRemote.create_with_canvas(FigureCanvasRemote, figure, 1),
        FigureManagerRemote,
    )

    # Concrete backend objects are accepted wherever the general API takes a
    # bare base type.
    figure.set_canvas(canvas)
    Artist().draw(RemoteRenderer())
    Event("x", canvas)
    key_press_handler(KeyEvent("key_press_event", canvas, "x"), canvas, toolbar)


def _check_remote_diamond_types(  # pragma: no cover
    qt_canvas: FigureCanvasQTRemote,
    qt_toolbar: NavigationToolbar2QTRemote,
    qt_manager: FigureManagerQTRemote,
    tk_canvas: FigureCanvasTkRemote,
    tk_toolbar: NavigationToolbar2TkRemote,
) -> None:
    # The first base wins for the cross-references.
    assert_type(qt_canvas.toolbar, NavigationToolbar2Remote | None)
    assert_type(qt_canvas.manager, FigureManagerRemote | None)
    assert_type(qt_toolbar.canvas, FigureCanvasRemote)
    assert_type(qt_manager.canvas, FigureCanvasRemote)
    assert_type(tk_canvas.toolbar, NavigationToolbar2Remote | None)
    assert_type(tk_toolbar.canvas, FigureCanvasRemote)
    # Toolkit-specific members remain available through the second base.
    qt_canvas.mouseEventCoords()
    tk_canvas.get_tk_widget()
    # ``NavigationToolbar2Tk.message`` is a ``tk.StringVar`` at runtime.
    tk_toolbar.message.set("x")


def _check_manager_type_propagation(  # pragma: no cover
    canvas_type: type[
        FigureCanvasBase[
            NavigationToolbar2[Any], SpecializedFigureManager, SpecializedTimer
        ]
    ],
    figure: Figure,
) -> None:
    assert_type(canvas_type.manager_class, type[SpecializedFigureManager])
    assert_type(canvas_type.new_manager(figure, 1), SpecializedFigureManager)


def _check_bare_generic_defaults(  # pragma: no cover
    canvas: FigureCanvasBase,
    manager: FigureManagerBase,
    toolbar: NavigationToolbar2[Any],
    renderer: RendererBase,
    agg: FigureCanvasAgg,
    remote_manager: FigureManagerBase[FigureCanvasRemote, NavigationToolbar2Remote],
) -> None:
    assert_type(
        remote_manager._toolbar2_class, type[NavigationToolbar2Remote] | None
    )
    # Bare base types degrade to ``Any`` specializations, not to the base
    # class itself, so they accept every concrete backend.
    assert_type(canvas.manager, FigureManagerBase[Any, Any] | None)
    assert_type(canvas.toolbar, NavigationToolbar2[Any] | None)
    assert_type(canvas.new_timer(), TimerBase)
    assert_type(manager.toolbar, NavigationToolbar2[Any] | ToolContainerBase | None)
    assert_type(renderer.new_gc(), GraphicsContextBase)
    assert_type(agg.manager, FigureManagerBase[Any, Any] | None)


def _check_non_gui_canvas_types(  # pragma: no cover
    pdf: FigureCanvasPdf,
    svg: FigureCanvasSVG,
    template: FigureCanvasTemplate,
    figure: Figure,
) -> None:
    assert_type(pdf.manager, FigureManagerBase[Any, Any] | None)
    assert_type(svg.manager, FigureManagerBase[Any, Any] | None)
    assert_type(template.manager, FigureManagerTemplate | None)
    assert_type(FigureCanvasTemplate.manager_class, type[FigureManagerTemplate])
    assert_type(FigureCanvasTemplate.new_manager(figure, 1), FigureManagerTemplate)


def _check_gui_canvas_types(  # pragma: no cover
    qt: FigureCanvasQT,
    qtagg: FigureCanvasQTAgg,
    qtcairo: FigureCanvasQTCairo,
    tk: FigureCanvasTk,
    tkagg: FigureCanvasTkAgg,
    tkcairo: FigureCanvasTkCairo,
    gtk3: FigureCanvasGTK3,
    gtk3agg: FigureCanvasGTK3Agg,
    gtk3cairo: FigureCanvasGTK3Cairo,
    gtk4: FigureCanvasGTK4,
    gtk4agg: FigureCanvasGTK4Agg,
    gtk4cairo: FigureCanvasGTK4Cairo,
    wx: _FigureCanvasWxBase,
    wxagg: FigureCanvasWxAgg,
    wxcairo: FigureCanvasWxCairo,
    figure: Figure,
) -> None:
    # Every combined "rendering + toolkit" canvas must retain the toolkit's
    # concrete manager, toolbar, and timer types even though the rendering
    # canvas comes first in the MRO.
    for qt_canvas in (qt, qtagg, qtcairo):
        assert_type(qt_canvas.manager, FigureManagerQT | None)
        assert_type(qt_canvas.toolbar, NavigationToolbar2QT | None)
        assert_type(qt_canvas.new_timer(), TimerQT)
        if qt_canvas.manager is not None:
            assert_type(qt_canvas.manager.canvas, FigureCanvasQT)
            assert_type(
                qt_canvas.manager.toolbar,
                NavigationToolbar2QT | ToolContainerBase | None,
            )
        if qt_canvas.toolbar is not None:
            assert_type(qt_canvas.toolbar.canvas, FigureCanvasQT)
    assert_type(FigureCanvasQTAgg.manager_class, type[FigureManagerQT])
    assert_type(FigureCanvasQTAgg.new_manager(figure, 1), FigureManagerQT)
    assert_type(FigureCanvasQTCairo.manager_class, type[FigureManagerQT])
    assert_type(qtagg.copy_from_bbox(figure.bbox), BufferRegion)

    for tk_canvas in (tk, tkagg, tkcairo):
        assert_type(tk_canvas.manager, FigureManagerTk | None)
        assert_type(tk_canvas.toolbar, NavigationToolbar2Tk | None)
        assert_type(tk_canvas.new_timer(), TimerTk)
    assert_type(FigureCanvasTkAgg.manager_class, type[FigureManagerTk])
    assert_type(FigureCanvasTkCairo.new_manager(figure, 1), FigureManagerTk)

    for gtk3_canvas in (gtk3, gtk3agg, gtk3cairo):
        assert_type(gtk3_canvas.manager, FigureManagerGTK3 | None)
        assert_type(gtk3_canvas.toolbar, NavigationToolbar2GTK3 | None)
        assert_type(gtk3_canvas.new_timer(), TimerGTK)
    assert_type(FigureCanvasGTK3Agg.manager_class, type[FigureManagerGTK3])

    for gtk4_canvas in (gtk4, gtk4agg, gtk4cairo):
        assert_type(gtk4_canvas.manager, FigureManagerGTK4 | None)
        assert_type(gtk4_canvas.toolbar, NavigationToolbar2GTK4 | None)
        assert_type(gtk4_canvas.new_timer(), TimerGTK)
    assert_type(FigureCanvasGTK4Cairo.manager_class, type[FigureManagerGTK4])

    for wx_canvas in (wx, wxagg, wxcairo):
        assert_type(wx_canvas.manager, FigureManagerWx | None)
        assert_type(wx_canvas.toolbar, NavigationToolbar2Wx | None)
        assert_type(wx_canvas.new_timer(), TimerWx)
    assert_type(FigureCanvasWxAgg.manager_class, type[FigureManagerWx])


def _check_web_canvas_types(  # pragma: no cover
    core_canvas: FigureCanvasWebAggCore,
    web_canvas: FigureCanvasWebAgg,
    nbagg_canvas: FigureCanvasNbAgg,
    figure: Figure,
) -> None:
    assert_type(core_canvas.manager, FigureManagerWebAggCore[Any] | None)
    assert_type(core_canvas.toolbar, NavigationToolbar2WebAgg | None)
    assert_type(core_canvas.new_timer(), TimerAsyncio)
    assert_type(core_canvas.get_diff_image(), bytes | None)

    assert_type(web_canvas.manager, FigureManagerWebAggApplication | None)
    assert_type(web_canvas.toolbar, NavigationToolbar2WebAgg | None)
    assert_type(FigureCanvasWebAgg.manager_class, type[FigureManagerWebAggApplication])
    assert_type(
        FigureCanvasWebAgg.new_manager(figure, 1), FigureManagerWebAggApplication
    )
    if web_canvas.manager is not None:
        assert_type(web_canvas.manager.ToolbarCls, type[NavigationToolbar2WebAgg])
        assert_type(
            web_canvas.manager.toolbar,
            NavigationToolbar2WebAgg | ToolContainerBase | None,
        )

    assert_type(nbagg_canvas.manager, FigureManagerNbAgg | None)
    assert_type(nbagg_canvas.toolbar, NavigationIPy | None)
    assert_type(nbagg_canvas.new_timer(), TimerAsyncio)
    assert_type(FigureCanvasNbAgg.manager_class, type[FigureManagerNbAgg])
    assert_type(FigureCanvasNbAgg.new_manager(figure, 1), FigureManagerNbAgg)
    if nbagg_canvas.manager is not None:
        assert_type(nbagg_canvas.manager.ToolbarCls, type[NavigationIPy])

    assert_type(FigureManagerWebAggCore.get_javascript(), str)
    assert_type(FigureManagerWebAggCore.get_javascript(io.StringIO()), None)
    assert_type(FigureManagerNbAgg.get_javascript(), str)
    assert_type(FigureManagerNbAgg.get_javascript(io.StringIO()), None)


def _check_agg_extension_types(  # pragma: no cover
    canvas: FigureCanvasAgg,
    renderer: RendererAgg,
    figure: Figure,
) -> None:
    region = canvas.copy_from_bbox(figure.bbox)
    assert_type(region, BufferRegion)
    assert_type(region.get_extents(), tuple[int, int, int, int])
    region.set_x(1)
    region.set_y(2)
    memoryview(region)
    canvas.restore_region(region)
    canvas.restore_region(region, bbox=(0.0, 0.0, 1.0, 1.0), xy=(0.0, 0.0))
    canvas.restore_region(region, bbox=Bbox.from_bounds(0, 0, 1, 1))
    assert_type(renderer.copy_from_bbox(figure.bbox), BufferRegion)
    renderer.restore_region(region)
    renderer.clear()
    renderer.start_filter()
    renderer.stop_filter(lambda image, dpi: (image, 0.0, 0.0))
    memoryview(renderer._renderer)


def _check_cairo_types(  # pragma: no cover
    canvas: FigureCanvasCairo,
    renderer: RendererCairo,
    figure: Figure,
) -> None:
    assert_type(canvas.manager, FigureManagerBase[Any, Any] | None)
    canvas.restore_region(canvas.copy_from_bbox(figure.bbox))
    Artist().draw(renderer)


def _check_negative_cases(  # pragma: no cover
    qt_canvas: FigureCanvasQTAgg,
    web_canvas: FigureCanvasWebAggCore,
    remote_manager: FigureManagerRemote,
    web_manager: FigureManagerWebAggCore[Any],
) -> None:
    # Cross-backend assignments must be rejected: the relationships are
    # invariant on purpose.
    qt_canvas.manager = remote_manager  # type: ignore[assignment]
    remote_manager.canvas = web_canvas  # type: ignore[assignment]
    # The Qt manager's toolbar slot only accepts Qt toolbars.
    if qt_canvas.manager is not None:
        qt_canvas.manager._toolbar2_class = NavigationToolbar2Remote  # type: ignore[assignment]
    # Return types that were previously over-narrow.
    web_manager.get_javascript().upper()
    web_manager.get_javascript(io.StringIO()).upper()  # type: ignore[attr-defined]
    web_canvas.mouseEventCoords()  # type: ignore[attr-defined]
    # The timer class must match the declared timer type parameter.
    FigureCanvasRemote._timer_cls = TimerBase  # type: ignore[assignment]


def _check_native_toolkit_members(  # pragma: no cover
    qt_canvas: FigureCanvasQTRemote,
    qt_toolbar: NavigationToolbar2QTRemote,
    tk_canvas: FigureCanvasTkRemote,
    tk_toolbar: NavigationToolbar2TkRemote,
) -> None:
    # GUI classes inherit their native toolkit widget at runtime.  For Qt, GTK
    # and wx the toolkit is untyped, so the stub spells that base ``Any`` and
    # the widget API is available (as ``Any``); Tk uses the real ``tkinter``
    # stubs.  A downstream thin client subclassing a GUI canvas relies on this.
    qt_canvas.show()
    qt_canvas.setFocus()
    assert_type(qt_canvas.width(), Any)
    assert_type(qt_canvas.devicePixelRatioF(), Any)
    assert_type(qt_toolbar._actions, dict[str, Any])
    qt_toolbar.sizeHint()
    # Matplotlib-owned members keep their precise types alongside ``Any``.
    assert_type(qt_canvas.figure, Figure)
    assert_type(qt_canvas.manager, FigureManagerRemote | None)
    assert_type(qt_toolbar.canvas, FigureCanvasRemote)

    assert_type(tk_canvas._tkcanvas, tk.Canvas)
    tk_canvas._tkcanvas.bind("<Configure>", lambda event: None)
    tk_canvas.get_tk_widget()
    tk_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    assert_type(tk_toolbar.winfo_width(), int)


def _check_renderer_base_boundaries(  # pragma: no cover
    renderer: RendererPdf,
    colorbar: Colorbar,
    figure: Figure,
    gridspec: GridSpec,
    image: AxesImage,
    indicator: InsetIndicator,
    offset_box: OffsetBox,
    cell: Cell,
    annotation: Annotation,
    offset_from: OffsetFrom,
) -> None:
    axes = colorbar.ax

    cell.get_window_extent(renderer)
    cell.get_tightbbox(renderer)
    cell.draw(renderer)
    axes.xaxis.get_tightbbox(renderer)
    axes.set_axes_locator(lambda ax, renderer: ax.bbox)
    axes.get_axes_locator()(axes, renderer)
    axes.get_tightbbox(renderer)
    colorbar.outline.get_window_extent(renderer)
    colorbar.outline.draw(renderer)
    figure.get_tightbbox(renderer)
    gridspec.tight_layout(figure, renderer)
    composite_images([image], renderer)
    image.make_image(renderer)
    image.draw(renderer)
    image.get_window_extent(renderer)
    indicator.draw(renderer)
    offset_box.get_bbox(renderer)
    offset_box.get_window_extent(renderer)
    bbox_artist(cell, renderer)
    draw_bbox(cell.get_window_extent(renderer), renderer)
    PathEffectRenderer([], renderer)
    cell.auto_set_font_size(renderer)
    cell.get_text_bounds(renderer)
    cell.get_required_width(renderer)
    TexManager.get_text_width_height_descent("", 10.0, renderer)
    text = cell.get_text()
    text.update_bbox_position_size(renderer)
    text.get_window_extent(renderer)
    offset_from(renderer)
    annotation.update_positions(renderer)
    annotation.get_window_extent(renderer)
    MixedModeRenderer(
        figure, 1.0, 1.0, 72.0, renderer, raster_renderer_class=RendererPdf
    )


def _check_canvas_base_boundaries(  # pragma: no cover
    canvas: SpecializedFigureCanvasAgg,
    renderer: RendererBase[Any],
    figure: Figure,
    subfigure: SubFigure,
    toolmanager: ToolManager,
    tool: ToolBase,
    selector: VertexSelector,
    draggable: DraggableBase,
    widget: AxesWidget,
    artist: Artist,
) -> None:
    assert_type(
        register_backend("backend_types_mypy", SpecializedFigureCanvasAgg), None
    )
    assert_type(
        get_registered_canvas_class("backend_types_mypy"),
        type[FigureCanvasBase[Any, Any, Any]],
    )

    event = Event("test_event", canvas)
    draw_event = DrawEvent("draw_event", canvas, renderer)
    resize_event = ResizeEvent("resize_event", canvas)
    close_event = CloseEvent("close_event", canvas)
    location_event = LocationEvent("motion_notify_event", canvas, 1, 2)
    mouse_event = MouseEvent("button_press_event", canvas, 1, 2)
    pick_event = PickEvent("pick_event", canvas, mouse_event, artist)
    key_event = KeyEvent("key_press_event", canvas, "x")

    events: list[Event] = [
        event,
        draw_event,
        resize_event,
        close_event,
        location_event,
        mouse_event,
        pick_event,
        key_event,
    ]
    for canvas_event in events:
        assert_type(canvas_event.canvas, FigureCanvasBase[Any, Any, Any])

    key_press_handler(key_event, canvas)
    button_press_handler(mouse_event, canvas)
    scroll_handler(mouse_event, canvas)

    assert_type(toolmanager.canvas, FigureCanvasBase[Any, Any, Any] | None)
    assert_type(tool.canvas, FigureCanvasBase[Any, Any, Any] | None)
    assert_type(
        _tool_registry,
        set[tuple[type[FigureCanvasBase[Any, Any, Any]], type[ToolBase]]],
    )
    assert_type(
        _register_tool_class(SpecializedFigureCanvasAgg, SpecializedTool),
        type[SpecializedTool],
    )
    assert_type(
        _register_tool_class(SpecializedFigureCanvasAgg)(SpecializedTool),
        type[SpecializedTool],
    )
    assert_type(
        _find_tool_class(SpecializedFigureCanvasAgg, SpecializedTool),
        type[SpecializedTool],
    )

    figure.set_canvas(canvas)
    assert_type(figure.canvas, FigureCanvasBase[Any, Any, Any])
    assert_type(subfigure.canvas, FigureCanvasBase[Any, Any, Any])
    assert_type(selector.canvas, FigureCanvasBase[Any, Any, Any])
    assert_type(draggable.canvas, FigureCanvasBase[Any, Any, Any])
    assert_type(widget.canvas, FigureCanvasBase[Any, Any, Any] | None)


# ---------------------------------------------------------------------------
# Runtime tests.
# ---------------------------------------------------------------------------


def test_backend_subclass_types() -> None:
    fig = Figure()
    canvas = SpecializedFigureCanvasAgg(fig)

    fig.set_canvas(canvas)
    assert_type(fig.canvas, FigureCanvasBase[Any, Any, Any])

    event = MouseEvent("button_press_event", canvas, 1, 2)
    assert event.canvas is canvas

    file_format = "backend_types"
    try:
        register_backend(file_format, SpecializedFigureCanvasAgg)
        assert get_registered_canvas_class(file_format) is SpecializedFigureCanvasAgg
    finally:
        vars(backend_bases)["_default_backends"].pop(file_format)
        vars(backend_bases)["_default_filetypes"].pop(file_format)

    pdf_file = PdfFile(io.BytesIO())
    try:
        renderer = RendererPdf(pdf_file, 72, 1, 1)
        Artist().draw(renderer)
    finally:
        pdf_file.close()


def test_remote_backend_runtime() -> None:
    # The generic bases are subscriptable at runtime (``__class_getitem__``),
    # so the class statements above import and instantiate.
    fig = Figure()
    canvas = FigureCanvasRemote(fig)
    manager = FigureManagerRemote(canvas, 1)
    assert canvas.manager is manager
    assert isinstance(manager.toolbar, NavigationToolbar2Remote)
    assert manager.toolbar.canvas is canvas
    assert isinstance(canvas.new_timer(), RemoteTimer)
    assert FigureCanvasRemote.manager_class is FigureManagerBase
    assert canvas._set_device_pixel_ratio(2.0)
    assert canvas.device_pixel_ratio == 2.0


def test_agg_buffer_region_runtime() -> None:
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    region = canvas.copy_from_bbox(fig.bbox)
    assert isinstance(region, BufferRegion)
    x0, y0, x1, y1 = region.get_extents()
    assert (x1 - x0, y1 - y0) == tuple(int(v) for v in fig.bbox.size)
    assert memoryview(region).ndim == 3
    canvas.restore_region(region)
    canvas.restore_region(region, bbox=(x0, y0, x1, y1), xy=(x0, y0))


def test_pdfpages_types(tmp_path: Path) -> None:
    fig = Figure()
    metadata: dict[str, str] = {"Title": "A title"}
    removed_metadata: dict[str, str | None] = {"Creator": None}

    with PdfPages(tmp_path / "output.pdf", metadata=metadata) as pdf:
        pdf.savefig(fig, dpi=100)
        info = pdf.infodict()
        info["ModDate"] = datetime.datetime.now()
        assert_type(info, dict[str, str | datetime.datetime | Name])
        assert info["Title"] == "A title"

    with PdfPages(tmp_path / "without-creator.pdf", metadata=removed_metadata) as pdf:
        pdf.savefig(fig)
        assert "Creator" not in pdf.infodict()

    assert FigureManagerPdf is FigureManagerBase
