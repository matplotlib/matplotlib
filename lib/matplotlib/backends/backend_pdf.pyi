import os
import types
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from enum import Enum
from functools import total_ordering
from typing import IO, Any, Literal, Protocol, Self, TypeAlias

import numpy as np
from _typeshed import ReadableBuffer, SupportsWrite
from numpy import typing as npt

from matplotlib import _api, path, transforms
from matplotlib._type1font import Type1Font
from matplotlib.backend_bases import FigureCanvasBase, GraphicsContextBase
from matplotlib.dviread import DviFont
from matplotlib.figure import Figure
from matplotlib.font_manager import FontPath, FontProperties
from matplotlib.text import Text
from matplotlib.transforms import BboxBase, Transform, TransformedBbox, TransformedPath
from matplotlib.typing import (
    CapStyleType,
    ColorType,
    JoinStyleType,
    LineStyleType,
    RGBColorType,
)

from . import _backend_pdf_ps

# XXX: Some of these might be worth moving to `mpl.typing`
_CommandType: TypeAlias = list[_SupportsPdfReprExt]
_CommandFuncType: TypeAlias = Callable[..., _CommandType]
_RectangleType: TypeAlias = tuple[float, float, float, float] | list[float]
# struct definition SketchParams in _backend_agg_basic_types.h
_SketchParamsType: TypeAlias = tuple[float, float, float]
_HatchType: TypeAlias = str
_HatchStyleType: TypeAlias = tuple[
    ColorType | None, ColorType | None, _HatchType | None, float
]

class _SupportsPdfRepr(Protocol):
    def pdfRepr(self) -> bytes: ...

_SupportsPdfReprExt: TypeAlias = (
    _SupportsPdfRepr
    | float
    | np.floating
    | bool
    | int
    | np.integer
    | str
    | bytes
    | dict[Name | bytes, _SupportsPdfReprExt]
    | list[_SupportsPdfReprExt]
    | tuple[_SupportsPdfReprExt, ...]
    | None
    | datetime
    | BboxBase
)

_MetadataDict: TypeAlias = dict[str, str | datetime | Name]

def pdfRepr(obj: _SupportsPdfReprExt) -> bytes: ...

class Reference:
    def __init__(self, id: int) -> None: ...
    def __repr__(self) -> str: ...
    def pdfRepr(self) -> bytes: ...
    def write(
        self, contents: _SupportsPdfReprExt, file: SupportsWrite[bytes]
    ) -> None: ...

@total_ordering
class Name:
    def __init__(self, name: Self | bytes | str) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def pdfRepr(self) -> bytes: ...

class Verbatim:
    def __init__(self, x: bytes) -> None: ...
    def pdfRepr(self) -> bytes: ...

class Op(Enum):
    close_fill_stroke = b"b"
    fill_stroke = b"B"
    fill = b"f"
    closepath = b"h"
    close_stroke = b"s"
    stroke = b"S"
    endpath = b"n"
    begin_text = b"BT"
    end_text = b"ET"
    curveto = b"c"
    rectangle = b"re"
    lineto = b"l"
    moveto = b"m"
    concat_matrix = b"cm"
    use_xobject = b"Do"
    setgray_stroke = b"G"
    setgray_nonstroke = b"g"
    setrgb_stroke = b"RG"
    setrgb_nonstroke = b"rg"
    setcolorspace_stroke = b"CS"
    setcolorspace_nonstroke = b"cs"
    setcolor_stroke = b"SCN"
    setcolor_nonstroke = b"scn"
    setdash = b"d"
    setlinejoin = b"j"
    setlinecap = b"J"
    setgstate = b"gs"
    gsave = b"q"
    grestore = b"Q"
    textpos = b"Td"
    selectfont = b"Tf"
    textmatrix = b"Tm"
    textrise = b"Ts"
    show = b"Tj"
    showkern = b"TJ"
    setlinewidth = b"w"
    clip = b"W"
    shading = b"sh"
    def pdfRepr(self) -> bytes: ...
    @classmethod
    def paint_path(cls, fill: bool, stroke: bool) -> bytes: ...

class Stream:
    def __init__(
        self,
        id: int,
        len: Reference | None,
        file: PdfFile,
        extra: dict[Name, Any] | None = None,
        png: dict[Any, Any] | None = None,
    ) -> None: ...
    def end(self) -> None: ...
    def write(self, data: bytes) -> None: ...

class PdfFile:
    def __init__(
        self,
        filename: str | os.PathLike | IO[Any],
        metadata: _MetadataDict | None = None,
    ) -> None: ...
    @property
    def dviFontInfo(self) -> dict[Name, types.SimpleNamespace]: ...
    def newPage(self, width: float, height: float) -> None: ...
    def newTextnote(
        self,
        text: _SupportsPdfReprExt,
        positionRect: _RectangleType = [-100, -100, 0, 0],
    ) -> None: ...
    def finalize(self) -> None: ...
    def close(self) -> None: ...
    def write(self, data: ReadableBuffer) -> None: ...
    def output(self, *data: _SupportsPdfReprExt) -> None: ...
    def beginStream(
        self,
        id: int,
        len: Reference | None,
        extra: dict[Name, Any] | None = None,
        png: dict[Any, Any] | None = None,
    ) -> None: ...
    def endStream(self) -> None: ...
    def outputStream(
        self, ref: Reference, data: bytes, *, extra: dict[Name, Any] | None = None
    ) -> None: ...
    def fontName(self, fontprop: FontPath | str, subset: int = 0) -> Name | None: ...
    def dviFontName(self, dvifont: DviFont) -> Name: ...
    def writeFonts(self) -> None: ...
    @_api.delete_parameter("3.11", "fontfile")
    def createType1Descriptor(
        self, t1font: Type1Font, fontfile: Any = None
    ) -> Reference: ...
    def embedTTF(
        self,
        filename: Iterable[str | bytes | os.PathLike | FontPath]
        | str
        | bytes
        | os.PathLike
        | FontPath,
        subset_index: int,
        charmap: dict[int, int],
    ) -> Reference: ...
    def alphaState(self, alpha: tuple[float, float]) -> Name: ...
    def writeExtGSTates(self) -> None: ...
    def hatchPattern(self, hatch_style: _HatchStyleType) -> Name: ...
    def writeHatches(self) -> None: ...
    def addGouraudTriangles(
        self, points: npt.ArrayLike, colors: npt.ArrayLike
    ) -> tuple[Name, Reference]: ...
    def writeGouraudTriangles(self) -> None: ...
    def imageObject(self, image: npt.NDArray[np.uint8]) -> Name: ...
    def writeImages(self) -> None: ...
    def markerObject(
        self,
        path: path.Path,
        trans: Transform,
        fill: bool,
        stroke: bool,
        lw: float,
        joinstyle: JoinStyleType,
        capstyle: CapStyleType,
    ) -> Name: ...
    def writeMarkers(self) -> None: ...
    def pathCollectionObject(
        self,
        gc: GraphicsContextBase,
        path: path.Path,
        trans: Transform,
        padding: float,
        filled: bool,
        stroked: bool,
    ) -> Name: ...
    def writePathCollectionTemplates(self) -> None: ...
    # types in _path.h::convert_to_string
    @staticmethod
    def pathOperations(
        path: path.Path,
        transform: Transform,
        clip: _RectangleType | None = None,
        simplify: bool | None = None,
        sketch: _SketchParamsType | None = None,
    ) -> list[Verbatim]: ...
    def writePath(
        self,
        path: path.Path,
        transform: Transform,
        clip: bool = False,
        sketch: _SketchParamsType | None = None,
    ) -> None: ...
    def reserveObject(self, name: str = "") -> Reference: ...
    def recordXref(self, id: int) -> None: ...
    def writeObject(
        self, object: _SupportsPdfReprExt, contents: dict[str, _SupportsPdfReprExt]
    ) -> None: ...
    def writeXref(self) -> None: ...
    def writeInfoDict(self) -> None: ...
    def writeTrailer(self) -> None: ...

class RendererPdf(_backend_pdf_ps.RendererPDFPSBase):
    paths: tuple[
        Name,
        path.Path,
        Transform,
        Reference,
        JoinStyleType,
        CapStyleType,
        float,
        bool,
        bool,
    ]
    def __init__(
        self, file: PdfFile, image_dpi: float, height: float, width: float
    ): ...
    def finalize(self) -> None: ...
    def check_gc(
        self, gc: GraphicsContextBase, fillcolor: ColorType | None = None
    ) -> None: ...
    def get_image_magnification(self) -> float: ...
    def draw_image(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        im: npt.ArrayLike,
        transform: transforms.Affine2DBase | None = None,
    ) -> None: ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        path: path.Path,
        transform: Transform,
        rgbFace: ColorType | None = None,
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[path.Path],
        all_transforms: Sequence[npt.ArrayLike],
        offsets: npt.ArrayLike | Sequence[npt.ArrayLike],
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
    # XXX: Here the implementation relies on `fill` and `stroke` which are not
    # in the interface of `GraphicsContextBase`. Here we use
    # `GraphicsContextPdf` to annotate `gc`, as a result, `RendererPdf` does not
    # strictly inherit from `RenderedBase` correctly.
    def draw_markers(
        self,
        gc: GraphicsContextPdf,  # type: ignore[override]
        marker_path: path.Path,
        marker_trans: Transform,
        path: path.Path,
        trans: Transform,
        rgbFace: ColorType | None = None,
    ) -> None: ...
    def draw_gouraud_triangles(
        self,
        gc: GraphicsContextBase,
        points: npt.ArrayLike,
        colors: npt.ArrayLike,
        trans: Transform,
    ) -> None: ...
    def draw_mathtext(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
    ) -> None: ...
    def draw_tex(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        *,
        mtext: Text | None = None,
    ) -> None: ...
    def encode_string(self, s: str, fonttype: int) -> bytes: ...
    def draw_text(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        ismath: bool | Literal["TeX"] = False,
        mtext: Text | None = None,
    ) -> None: ...
    def new_gc(self) -> GraphicsContextPdf: ...

class GraphicsContextPdf(GraphicsContextBase):
    file: PdfFile
    capstyles: dict[CapStyleType, int]
    joinstyles: dict[JoinStyleType, int]
    commands: tuple[tuple[str, ...], _CommandFuncType]
    def __init__(self, file: PdfFile): ...
    def __repr__(self) -> str: ...
    def stroke(self) -> bool: ...
    def fill(self, *args: ColorType) -> bool: ...
    def paint(self) -> Op: ...
    def capstyle_cmd(self, style: CapStyleType) -> _CommandType: ...
    def joinstyle_cmd(self, style: JoinStyleType) -> _CommandType: ...
    def linewidth_cmd(self, width: float) -> _CommandType: ...
    def dash_cmd(self, dashes: tuple[float, Sequence[float]]) -> _CommandType: ...
    def alpha_cmd(
        self,
        alpha: tuple[float, float],
        forced: bool,
        effective_alphas: tuple[float, float],
    ) -> _CommandType: ...
    def hatch_cmd(
        self, hatch: _HatchType, hatch_color: ColorType, hatch_linewidth: float
    ) -> _CommandType: ...
    def rgb_cmd(self, rgb: RGBColorType) -> _CommandType: ...
    def fillcolor_cmd(self, rgb: RGBColorType) -> _CommandType: ...
    def push(self) -> list[Op]: ...
    def pop(self) -> list[Op]: ...
    def clip_cmd(
        self, cliprect: TransformedBbox, clippath: TransformedPath
    ) -> _CommandType: ...
    def delta(self, other: GraphicsContextBase) -> _CommandType: ...
    def copy_properties(self, other: GraphicsContextBase) -> None: ...
    def finalize(self) -> list[Op]: ...

class PdfPages:
    def __init__(
        self,
        filename: str | os.PathLike | IO[Any],
        keep_empty: None = None,
        metadata: _MetadataDict | None = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: object, exc_tb: object
    ) -> None: ...
    def close(self) -> None: ...
    def infodict(self) -> _MetadataDict: ...
    def savefig(
        self, figure: Figure | int | None = None, **kwargs: dict[str, Any]
    ) -> None: ...
    def get_pagecount(self) -> int: ...
    def attach_note(
        self,
        text: _SupportsPdfReprExt,
        positionRect: _RectangleType = [-100, -100, 0, 0],
    ) -> None: ...

class FigureCanvasPdf(FigureCanvasBase):
    filetypes: dict[str, str]
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def print_pdf(
        self,
        filename: PdfPages | str | os.PathLike | IO[Any],
        *,
        bbox_inches_restore: _RectangleType | None = None,
        metadata: _MetadataDict | None = None,
    ) -> None: ...
    def draw(self) -> None: ...
