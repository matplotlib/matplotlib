import contextlib
from collections.abc import Callable
from typing import Any, TypeVar, overload
from typing_extensions import ParamSpec, Unpack

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


class MatplotlibDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning class."""


class DeprecationKwargs(dict):
    message: str
    alternative: str
    pending: bool
    obj_type: str
    addendum: str
    removal: str


class NamedDeprecationKwargs(DeprecationKwargs):
    name: str


def warn_deprecated(since: str, **kwargs: NamedDeprecationKwargs) -> None:
    """Warn about deprecated features."""


def deprecated(since: str, **kwargs: NamedDeprecationKwargs) -> Callable[[_T], _T]:
    """Decorator to mark a function or method as deprecated."""


class deprecate_privatize_attribute(Any):
    def __init__(self, since: str, **kwargs: NamedDeprecationKwargs):
        """Deprecate and privatize an attribute."""


    def __set_name__(self, owner: type[object], name: str) -> None:
        """Set the name of the owner class."""


DECORATORS = {}


@overload
def rename_parameter(
    since: str, old: str, new: str, func: None = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


@overload
def rename_parameter(
    since: str, old: str, new: str, func: Callable[_P, _R]
) -> Callable[_P, _R]:
    ...


class _deprecated_parameter_class:
    ...


_deprecated_parameter = _deprecated_parameter_class()


@overload
def delete_parameter(
    since: str, name: str, func: None = ..., **kwargs: NamedDeprecationKwargs
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


@overload
def delete_parameter(
    since: str, name: str, func: Callable[_P, _R], **kwargs: NamedDeprecationKwargs
) -> Callable[_P, _R]:
    ...


@overload
def make_keyword_only(
    since: str, name: str, func: None = ...
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


@overload
def make_keyword_only(
    since: str, name: str, func: Callable[_P, _R]
) -> Callable[_P, _R]:
    ...


def deprecate_method_override(
    method: Callable[_P, _R],
    obj: object | type,
    *,
    allow_empty: bool = ...,
    since: str,
    **kwargs: NamedDeprecationKwargs
) -> Callable[_P, _R]:
    """Deprecate a method override."""


@contextlib.contextmanager
def suppress_matplotlib_deprecation_warning() -> None:
    """Suppress Matplotlib deprecation warnings within a context."""
