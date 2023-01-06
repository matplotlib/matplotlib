from matplotlib.backend_bases import RendererBase, GraphicsContextBase
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.transforms import Transform

from typing import Iterable, Sequence
from matplotlib._typing import Color

class AbstractPathEffect:
    def __init__(self, offset: tuple[float, float] = ...) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: Color | None = ...,
    ) -> None: ...

class PathEffectRenderer(RendererBase):
    def __init__(
        self, path_effects: Iterable[AbstractPathEffect], renderer: RendererBase
    ) -> None: ...
    def copy_with_path_effect(self, path_effects: Iterable[AbstractPathEffect]): ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: Color | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path: Path,
        marker_trans: Transform,
        path: Path,
        *args,
        **kwargs
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[Path],
        *args,
        **kwargs
    ) -> None: ...
    def __getattribute__(self, name: str): ...

class Normal(AbstractPathEffect): ...

class Stroke(AbstractPathEffect):
    def __init__(self, offset: tuple[float, float] = ..., **kwargs) -> None: ...
    # rgbFace becomes non-optional
    def draw_path(self, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: Color = ...) -> None: ...  # type: ignore

class withStroke(Stroke): ...

class SimplePatchShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_rgbFace: Color | None = ...,
        alpha: float | None = ...,
        rho: float = ...,
        **kwargs
    ) -> None: ...
    # rgbFace becomes non-optional
    def draw_path(self, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: Color = ...) -> None: ...  # type: ignore

class withSimplePatchShadow(SimplePatchShadow): ...

class SimpleLineShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_color: Color = ...,
        alpha: float = ...,
        rho: float = ...,
        **kwargs
    ) -> None: ...
    # rgbFace becomes non-optional
    def draw_path(self, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: Color = ...) -> None: ...  # type: ignore

class PathPatchEffect(AbstractPathEffect):
    patch: Patch
    def __init__(self, offset: tuple[float, float] = ..., **kwargs) -> None: ...
    # rgbFace becomes non-optional
    def draw_path(self, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: Color = ...) -> None: ...  # type: ignore

class TickedStroke(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        spacing: float = ...,
        angle: float = ...,
        length: float = ...,
        **kwargs
    ) -> None: ...
    # rgbFace becomes non-optional
    def draw_path(self, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: Color = ...) -> None: ...  # type: ignore

class withTickedStroke(TickedStroke): ...
