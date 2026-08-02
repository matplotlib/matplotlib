import datetime
import io
from pathlib import Path
from typing import Any, assert_type

import matplotlib.backend_bases as backend_bases
from matplotlib.artist import Artist
from matplotlib.backend_bases import (
    CloseEvent,
    DrawEvent,
    Event,
    FigureCanvasBase,
    FigureManagerBase,
    KeyEvent,
    LocationEvent,
    MouseEvent,
    NavigationToolbar2,
    PickEvent,
    ResizeEvent,
    RendererBase,
    TimerBase,
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage, composite_images
from matplotlib.inset import InsetIndicator
from matplotlib.lines import VertexSelector
from matplotlib.offsetbox import DraggableBase, OffsetBox
from matplotlib.patches import bbox_artist, draw_bbox
from matplotlib.patheffects import PathEffectRenderer
from matplotlib.table import Cell
from matplotlib.texmanager import TexManager
from matplotlib.text import Annotation, OffsetFrom
from matplotlib.widgets import AxesWidget
from matplotlib.backends.backend_pdf import FigureManagerPdf, Name, PdfPages


class SpecializedTimer(TimerBase):
    pass


class SpecializedFigureCanvasAgg(
    FigureCanvasAgg[NavigationToolbar2[Any], FigureManagerBase[Any], SpecializedTimer]
):
    pass


class SpecializedTool(ToolBase):
    pass


def _check_renderer_base_boundaries(
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
    # This helper is checked by mypy but not executed by pytest.
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


def _check_canvas_base_boundaries(
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
    # This helper is checked by mypy but not executed by pytest.
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

    for canvas_event in [
        event,
        draw_event,
        resize_event,
        close_event,
        location_event,
        mouse_event,
        pick_event,
        key_event,
    ]:
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
