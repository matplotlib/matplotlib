from typing import cast
from enum import Enum

class _AutoStringNameEnum(Enum):
    def __hash__(self) -> int: ...

class JoinStyle(str, _AutoStringNameEnum):
    miter = cast(str, ...)
    round = cast(str, ...)
    bevel = cast(str, ...)
    @staticmethod
    def demo() -> None: ...

class CapStyle(str, _AutoStringNameEnum):
    butt = cast(str, ...)
    projecting = cast(str, ...)
    round = cast(str, ...)
    @staticmethod
    def demo() -> None: ...
