from collections.abc import Callable
import contextlib
from typing import Any, Literal, TypedDict, Unpack, overload

class MatplotlibDeprecationWarning(DeprecationWarning): ...

class DeprecationKwargs(TypedDict, total=False):
    message: str
    alternative: str
    pending: bool
    obj_type: str
    addendum: str
    removal: str | Literal[False]

class NamedDeprecationKwargs(DeprecationKwargs, total=False):
    name: str

def warn_deprecated(since: str, **kwargs: Unpack[NamedDeprecationKwargs]) -> None: ...
def deprecated[T](
    since: str, **kwargs: Unpack[NamedDeprecationKwargs]
) -> Callable[[T], T]: ...

class deprecate_privatize_attribute(Any):
    def __init__(self, since: str, **kwargs: Unpack[NamedDeprecationKwargs]): ...
    def __set_name__(self, owner: type[object], name: str) -> None: ...

DECORATORS: dict[Callable, Callable] = ...

@overload
def rename_parameter[**P, R](
    since: str, old: str, new: str, func: None = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def rename_parameter[**P, R](
    since: str, old: str, new: str, func: Callable[P, R]
) -> Callable[P, R]: ...

class _deprecated_parameter_class: ...

_deprecated_parameter: _deprecated_parameter_class

@overload
def delete_parameter[**P, R](
    since: str, name: str, func: None = ..., **kwargs: Unpack[DeprecationKwargs]
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def delete_parameter[**P, R](
    since: str, name: str, func: Callable[P, R], **kwargs: Unpack[DeprecationKwargs]
) -> Callable[P, R]: ...
@overload
def make_keyword_only[**P, R](
    since: str, name: str, func: None = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def make_keyword_only[**P, R](
    since: str, name: str, func: Callable[P, R]
) -> Callable[P, R]: ...
def deprecate_method_override[**P, R](
    method: Callable[P, R],
    obj: object | type,
    *,
    allow_empty: bool = ...,
    since: str,
    **kwargs: Unpack[NamedDeprecationKwargs]
) -> Callable[P, R]: ...
def suppress_matplotlib_deprecation_warning() -> (
    contextlib.AbstractContextManager[None]
): ...
