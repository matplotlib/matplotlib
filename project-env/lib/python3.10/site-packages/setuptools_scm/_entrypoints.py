from __future__ import annotations

import warnings
from typing import Any
from typing import Iterator
from typing import overload
from typing import TYPE_CHECKING

from .utils import function_has_arg
from .utils import trace
from .version import ScmVersion

if TYPE_CHECKING:
    from .config import Configuration
    from typing_extensions import Protocol
    from . import _types as _t
else:
    Configuration = Any

    class Protocol:
        pass


class MaybeConfigFunction(Protocol):
    __name__: str

    @overload
    def __call__(self, root: _t.PathT, config: Configuration) -> ScmVersion | None:
        pass

    @overload
    def __call__(self, root: _t.PathT) -> ScmVersion | None:
        pass


def _call_entrypoint_fn(
    root: _t.PathT, config: Configuration, fn: MaybeConfigFunction
) -> ScmVersion | None:
    if function_has_arg(fn, "config"):
        return fn(root, config=config)
    else:
        warnings.warn(
            f"parse function {fn.__module__}.{fn.__name__}"
            " are required to provide a named argument"
            " 'config', setuptools_scm>=8.0 will remove support.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return fn(root)


def _version_from_entrypoints(
    config: Configuration, fallback: bool = False
) -> ScmVersion | None:
    if fallback:
        entrypoint = "setuptools_scm.parse_scm_fallback"
        root = config.fallback_root
    else:
        entrypoint = "setuptools_scm.parse_scm"
        root = config.absolute_root

    from .discover import iter_matching_entrypoints

    trace("version_from_ep", entrypoint, root)
    for ep in iter_matching_entrypoints(root, entrypoint, config):
        version: ScmVersion | None = _call_entrypoint_fn(root, config, ep.load())
        trace(ep, version)
        if version:
            return version
    return None


try:
    from importlib.metadata import entry_points  # type: ignore
except ImportError:
    try:
        from importlib_metadata import entry_points
    except ImportError:
        from collections import defaultdict

        def entry_points() -> dict[str, list[_t.EntrypointProtocol]]:
            warnings.warn(
                "importlib metadata missing, "
                "this may happen at build time for python3.7"
            )
            return defaultdict(list)


def iter_entry_points(
    group: str, name: str | None = None
) -> Iterator[_t.EntrypointProtocol]:
    all_eps = entry_points()
    if hasattr(all_eps, "select"):
        eps = all_eps.select(group=group)
    else:
        eps = all_eps[group]
    if name is None:
        return iter(eps)
    return (ep for ep in eps if ep.name == name)
