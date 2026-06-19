from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.typing import RcStyleType

def remove_ticks_and_titles(figure: Figure) -> None: ...
def image_comparison[**P, R](
    baseline_images: list[str] | None,
    extensions: list[str] | None = ...,
    tol: float = ...,
    freetype_version: tuple[str, str] | str | None = ...,
    remove_text: bool = ...,
    savefig_kwarg: dict[str, Any] | None = ...,
    style: RcStyleType | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def check_figures_equal[**P, R](
    *, extensions: Sequence[str] = ..., tol: float = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def _image_directories(func: Callable) -> tuple[Path, Path]: ...
