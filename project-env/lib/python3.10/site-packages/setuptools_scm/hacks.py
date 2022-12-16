from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import _types as _t
from .config import Configuration
from .utils import data_from_mime
from .utils import trace
from .version import meta
from .version import ScmVersion
from .version import tag_to_version

_UNKNOWN = "UNKNOWN"


def parse_pkginfo(
    root: _t.PathT, config: Configuration | None = None
) -> ScmVersion | None:

    pkginfo = os.path.join(root, "PKG-INFO")
    trace("pkginfo", pkginfo)
    data = data_from_mime(pkginfo)
    version = data.get("Version", _UNKNOWN)
    if version != _UNKNOWN:
        return meta(version, preformatted=True, config=config)
    else:
        return None


def parse_pip_egg_info(
    root: _t.PathT, config: Configuration | None = None
) -> ScmVersion | None:
    pipdir = os.path.join(root, "pip-egg-info")
    if not os.path.isdir(pipdir):
        return None
    items = os.listdir(pipdir)
    trace("pip-egg-info", pipdir, items)
    if not items:
        return None
    return parse_pkginfo(os.path.join(pipdir, items[0]), config=config)


def fallback_version(root: _t.PathT, config: Configuration) -> ScmVersion | None:
    if config.parentdir_prefix_version is not None:
        _, parent_name = os.path.split(os.path.abspath(root))
        if parent_name.startswith(config.parentdir_prefix_version):
            version = tag_to_version(
                parent_name[len(config.parentdir_prefix_version) :], config
            )
            if version is not None:
                return meta(str(version), preformatted=True, config=config)
    if config.fallback_version is not None:
        trace("FALLBACK")
        return meta(config.fallback_version, preformatted=True, config=config)
    return None
