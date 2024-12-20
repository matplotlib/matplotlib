from typing import cast
from enum import Enum


class JoinStyle(str, Enum):
    miter = cast(str, ...)
    round = cast(str, ...)
    bevel = cast(str, ...)
    @staticmethod
    def demo() -> None: ...


class CapStyle(str, Enum):
    butt = cast(str, ...)
    projecting = cast(str, ...)
    round = cast(str, ...)
    @staticmethod
    def demo() -> None: ...
