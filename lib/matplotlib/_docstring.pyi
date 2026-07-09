from collections.abc import Callable
from typing import Any, overload


def kwarg_doc[T](text: str) -> Callable[[T], T]: ...


class Substitution:
    @overload
    def __init__(self, *args: str): ...
    @overload
    def __init__(self, **kwargs: str): ...
    def __call__[T](self, func: T) -> T: ...


class _ArtistKwdocLoader(dict[str, str]):
    def __missing__(self, key: str) -> str: ...


class _ArtistPropertiesSubstitution:
    def __init__(self) -> None: ...
    def register(self, **kwargs) -> None: ...
    def __call__[T](self, obj: T) -> T: ...


def copy[T](source: Any) -> Callable[[T], T]: ...


dedent_interpd: _ArtistPropertiesSubstitution
interpd: _ArtistPropertiesSubstitution
