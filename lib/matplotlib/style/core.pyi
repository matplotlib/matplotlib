from collections.abc import Generator
from typing import Any
from pathlib import Path
import contextlib

from matplotlib import RcParams

StyleType = str | dict[str, Any] | Path | list[str | Path | dict[str, Any]]

USER_LIBRARY_PATHS: list[str] = ...
STYLE_EXTENSION: str = ...

def use(style: StyleType) -> None: ...
@contextlib.contextmanager
def context(
    style: StyleType, after_reset: bool = ...
) -> Generator[None, None, None]: ...

class _StyleLibrary(dict[str, RcParams]): ...

library: _StyleLibrary
available: list[str]

def reload_library() -> None: ...
