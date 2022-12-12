from __future__ import annotations

import os
import warnings
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import setuptools

from . import _get_version
from . import _version_missing
from ._entrypoints import iter_entry_points
from ._integration.setuptools import (
    read_dist_name_from_setup_cfg as _read_dist_name_from_setup_cfg,
)
from .config import Configuration
from .utils import do
from .utils import trace

if TYPE_CHECKING:
    from . import _types as _t


def _warn_on_old_setuptools(_version: str = setuptools.__version__) -> None:
    if int(_version.split(".")[0]) < 45:
        warnings.warn(
            RuntimeWarning(
                f"""
ERROR: setuptools=={_version} is used in combination with setuptools_scm>=6.x

Your build configuration is incomplete and previously worked by accident!
setuptools_scm requires setuptools>=45


This happens as setuptools is unable to replace itself when a activated build dependency
requires a more recent setuptools version
(it does not respect "setuptools>X" in setup_requires).


setuptools>=31 is required for setup.cfg metadata support
setuptools>=42 is required for pyproject.toml configuration support

Suggested workarounds if applicable:
 - preinstalling build dependencies like setuptools_scm before running setup.py
 - installing setuptools_scm using the system package manager to ensure consistency
 - migrating from the deprecated setup_requires mechanism to pep517/518
   and using a pyproject.toml to declare build dependencies
   which are reliably pre-installed before running the build tools
"""
            )
        )


_warn_on_old_setuptools()


def _assign_version(dist: setuptools.Distribution, config: Configuration) -> None:
    maybe_version = _get_version(config)

    if maybe_version is None:
        _version_missing(config)
    else:
        dist.metadata.version = maybe_version


def version_keyword(
    dist: setuptools.Distribution,
    keyword: str,
    value: bool | dict[str, Any] | Callable[[], dict[str, Any]],
) -> None:
    if not value:
        return
    elif value is True:
        value = {}
    elif callable(value):
        value = value()
    assert (
        "dist_name" not in value
    ), "dist_name may not be specified in the setup keyword "

    trace(
        "version keyword",
        vars(dist.metadata),
    )
    dist_name = dist.metadata.name  # type: str | None
    if dist_name is None:
        dist_name = _read_dist_name_from_setup_cfg()
    config = Configuration(dist_name=dist_name, **value)
    _assign_version(dist, config)


def find_files(path: _t.PathT = "") -> list[str]:
    for ep in iter_entry_points("setuptools_scm.files_command"):
        command = ep.load()
        if isinstance(command, str):
            # this technique is deprecated
            res = do(ep.load(), path or ".").splitlines()
        else:
            res = command(path)
        if res:
            return res
    return []


def infer_version(dist: setuptools.Distribution) -> None:
    trace(
        "finalize hook",
        vars(dist.metadata),
    )
    dist_name = dist.metadata.name
    if dist_name is None:
        dist_name = _read_dist_name_from_setup_cfg()
    if not os.path.isfile("pyproject.toml"):
        return
    if dist_name == "setuptools_scm":
        return
    try:
        config = Configuration.from_file(dist_name=dist_name)
    except LookupError as e:
        trace(e)
    else:
        _assign_version(dist, config)
