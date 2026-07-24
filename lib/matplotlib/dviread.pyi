import dataclasses
from pathlib import Path
import io
import os
from enum import Enum
from collections.abc import Generator

from typing import NamedTuple, Self, Literal, Iterable

from .ft2font import CharacterCodeType, GlyphIndexType


class _dvistate(Enum):
    pre = ...
    outer = ...
    inpage = ...
    post = ...
    post_post = ...
    finale = ...

class Ops:
    class Op(NamedTuple):
        code: int
        name: str
        args: dict

    @dataclasses.dataclass(slots=True)
    class DispatchTable:
        entries: list = []

        def op(self, bmin: int, bmax: int, opname: str,
            arg_types: str ='', arg_names: str ='', extra=None): ...
        def __enter__(self) -> Self: ...
        def __exit__(self, *exc) -> Literal[False]: ...


    @classmethod
    def read_op(cls, f, table: DispatchTable) -> Generator[Op, None, None]: ...

    @classmethod
    def read_io(cls, f, table: DispatchTable | None = None) -> Generator[Op, None, None]: ...

    @classmethod
    def read_file(cls, filename: str, **kwargs) -> Generator[Op, None, None]: ...

    @classmethod
    def read_bytes(cls, b: bytes, **kwargs) -> Generator[Op, None, None]: ...

    tbl_dvi: DispatchTable
    tbl_vf_outer: DispatchTable
    tbl_vf_inner: DispatchTable

class Page(NamedTuple):
    text: list[Text]
    boxes: list[Box]
    height: int
    width: int
    descent: int

@dataclasses.dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    height: int
    width: int
    color: str | None = None

    def __iter__(self) -> Iterable[int]: ...
    def __getitem__(self, i: int) -> int: ...
    def replace(self, /, **kwargs) -> Self: ...

@dataclasses.dataclass(frozen=True, slots=True)
class Text:
    x: int
    y: int
    font: DviFont
    glyph: CharacterCodeType
    width: int
    color: str | None = None

    @property
    def font_path(self) -> Path: ...
    @property
    def font_size(self) -> float: ...
    @property
    def font_effects(self) -> dict[str, float]: ...
    @property
    def index(self) -> GlyphIndexType: ...  # type: ignore[override]
    @property
    def glyph_name_or_index(self) -> GlyphIndexType | str: ...

    def __iter__(self) -> Iterable: ...
    def __getitem__(self, i: int) -> int | DviFont: ...
    def replace(self, /, **kwargs) -> Self: ...

@dataclasses.dataclass(slots=True)
class VM:
    stack: list = []
    text: list[Text] = []
    boxes: list[Box] = []
    colors: list[str] = []
    down_stack: list[int] = []
    fonts: dict = {}
    state: _dvistate = _dvistate.pre
    baseline_v: None | int = None
    h: int = 0
    v: int = 0
    w: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    f: int = 0

    @property
    def color(self) -> str | None: ...

class Dvi:
    file: io.BufferedReader
    dpi: float | None
    fonts: dict[int, DviFont]
    state: _dvistate
    def __init__(self, filename: str | os.PathLike, dpi: float | None) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, etype, evalue, etrace) -> None: ...
    def __iter__(self) -> Generator[Page, None, None]: ...
    def close(self) -> None: ...

class DviFont:
    texname: bytes
    def __init__(
        self, scale: float, metrics: Tfm | TtfMetrics, texname: bytes, vf: Vf | None
    ) -> None: ...
    @classmethod
    def from_luatex(cls, scale: float, texname: bytes) -> DviFont: ...
    @classmethod
    def from_xetex(
        cls, scale: float, texname: bytes, subfont: int, effects: dict[str, float]
    ) -> DviFont: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def size(self) -> float: ...
    @property
    def widths(self) -> list[int]: ...
    @property
    def fname(self) -> str: ...
    @property
    def face_index(self) -> int: ...
    def resolve_path(self) -> Path: ...
    @property
    def subfont(self) -> int: ...
    @property
    def effects(self) -> dict[str, float]: ...

class Vf(Dvi):
    def __init__(self, filename: str | os.PathLike) -> None: ...
    def __getitem__(self, code: int) -> Page: ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class TexMetrics:
    tex_width: int
    tex_height: int
    tex_depth: int
    # work around mypy not respecting kw_only=True in stub files
    __match_args__ = ()

class Tfm:
    checksum: int
    design_size: int
    def __init__(self, filename: str | os.PathLike) -> None: ...
    def get_metrics(self, idx: int) -> TexMetrics | None: ...
    @property
    def width(self) -> dict[int, int]: ...
    @property
    def height(self) -> dict[int, int]: ...
    @property
    def depth(self) -> dict[int, int]: ...

class TtfMetrics:
    def __init__(self, filename: str | os.PathLike) -> None: ...
    def get_metrics(self, idx: int) -> TexMetrics: ...

class PsFont(NamedTuple):
    texname: bytes
    psname: bytes
    effects: dict[str, float]
    encoding: None | bytes
    filename: str

class PsfontsMap:
    def __new__(cls, filename: str | os.PathLike) -> Self: ...
    def __getitem__(self, texname: bytes) -> PsFont: ...

def find_tex_file(filename: str | os.PathLike) -> str: ...
