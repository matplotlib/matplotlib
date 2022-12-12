""" configuration """
from __future__ import annotations

import os
import re
import warnings
from typing import Any
from typing import Callable
from typing import cast
from typing import Pattern
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

from ._integration.pyproject_reading import (
    get_args_for_pyproject as _get_args_for_pyproject,
)
from ._integration.pyproject_reading import read_pyproject as _read_pyproject
from ._version_cls import NonNormalizedVersion
from ._version_cls import Version
from .utils import trace


if TYPE_CHECKING:
    from . import _types as _t
    from setuptools_scm.version import ScmVersion

DEFAULT_TAG_REGEX = r"^(?:[\w-]+-)?(?P<version>[vV]?\d+(?:\.\d+){0,2}[^\+]*)(?:\+.*)?$"
DEFAULT_VERSION_SCHEME = "guess-next-dev"
DEFAULT_LOCAL_SCHEME = "node-and-date"


def _check_tag_regex(value: str | Pattern[str] | None) -> Pattern[str]:
    if not value:
        value = DEFAULT_TAG_REGEX
    regex = re.compile(value)

    group_names = regex.groupindex.keys()
    if regex.groups == 0 or (regex.groups > 1 and "version" not in group_names):
        warnings.warn(
            "Expected tag_regex to contain a single match group or a group named"
            " 'version' to identify the version part of any tag."
        )

    return regex


def _check_absolute_root(root: _t.PathT, relative_to: _t.PathT | None) -> str:
    trace("abs root", repr(locals()))
    if relative_to:
        if (
            os.path.isabs(root)
            and os.path.isabs(relative_to)
            and not os.path.commonpath([root, relative_to]) == root
        ):
            warnings.warn(
                "absolute root path '%s' overrides relative_to '%s'"
                % (root, relative_to)
            )
        if os.path.isdir(relative_to):
            warnings.warn(
                "relative_to is expected to be a file,"
                " its the directory %r\n"
                "assuming the parent directory was passed" % (relative_to,)
            )
            trace("dir", relative_to)
            root = os.path.join(relative_to, root)
        else:
            trace("file", relative_to)
            root = os.path.join(os.path.dirname(relative_to), root)
    return os.path.abspath(root)


_VersionT = Union[Version, NonNormalizedVersion]


def _validate_version_cls(
    version_cls: type[_VersionT] | str | None, normalize: bool
) -> type[_VersionT]:
    if not normalize:
        # `normalize = False` means `version_cls = NonNormalizedVersion`
        if version_cls is not None:
            raise ValueError(
                "Providing a custom `version_cls` is not permitted when "
                "`normalize=False`"
            )
        return NonNormalizedVersion
    else:
        # Use `version_cls` if provided, default to packaging or pkg_resources
        if version_cls is None:
            return Version
        elif isinstance(version_cls, str):
            try:
                # Not sure this will work in old python
                import importlib

                pkg, cls_name = version_cls.rsplit(".", 1)
                version_cls_host = importlib.import_module(pkg)
                return cast(Type[_VersionT], getattr(version_cls_host, cls_name))
            except:  # noqa
                raise ValueError(f"Unable to import version_cls='{version_cls}'")
        else:
            return version_cls


class Configuration:
    """Global configuration model"""

    parent: _t.PathT | None
    _root: str
    _relative_to: str | None
    version_cls: type[_VersionT]

    def __init__(
        self,
        relative_to: _t.PathT | None = None,
        root: _t.PathT = ".",
        version_scheme: (
            str | Callable[[ScmVersion], str | None]
        ) = DEFAULT_VERSION_SCHEME,
        local_scheme: (str | Callable[[ScmVersion], str | None]) = DEFAULT_LOCAL_SCHEME,
        write_to: _t.PathT | None = None,
        write_to_template: str | None = None,
        tag_regex: str | Pattern[str] = DEFAULT_TAG_REGEX,
        parentdir_prefix_version: str | None = None,
        fallback_version: str | None = None,
        fallback_root: _t.PathT = ".",
        parse: Any | None = None,
        git_describe_command: _t.CMD_TYPE | None = None,
        dist_name: str | None = None,
        version_cls: type[_VersionT] | type | str | None = None,
        normalize: bool = True,
        search_parent_directories: bool = False,
    ):
        # TODO:
        self._relative_to = None if relative_to is None else os.fspath(relative_to)
        self._root = "."

        self.root = os.fspath(root)
        self.version_scheme = version_scheme
        self.local_scheme = local_scheme
        self.write_to = write_to
        self.write_to_template = write_to_template
        self.parentdir_prefix_version = parentdir_prefix_version
        self.fallback_version = fallback_version
        self.fallback_root = fallback_root  # type: ignore
        self.parse = parse
        self.tag_regex = tag_regex  # type: ignore
        self.git_describe_command = git_describe_command
        self.dist_name = dist_name
        self.search_parent_directories = search_parent_directories
        self.parent = None

        self.version_cls = _validate_version_cls(version_cls, normalize)

    @property
    def fallback_root(self) -> str:
        return self._fallback_root

    @fallback_root.setter
    def fallback_root(self, value: _t.PathT) -> None:
        self._fallback_root = os.path.abspath(value)

    @property
    def absolute_root(self) -> str:
        return self._absolute_root

    @property
    def relative_to(self) -> str | None:
        return self._relative_to

    @relative_to.setter
    def relative_to(self, value: _t.PathT) -> None:
        self._absolute_root = _check_absolute_root(self._root, value)
        self._relative_to = os.fspath(value)
        trace("root", repr(self._absolute_root))
        trace("relative_to", repr(value))

    @property
    def root(self) -> str:
        return self._root

    @root.setter
    def root(self, value: _t.PathT) -> None:
        self._absolute_root = _check_absolute_root(value, self._relative_to)
        self._root = os.fspath(value)
        trace("root", repr(self._absolute_root))
        trace("relative_to", repr(self._relative_to))

    @property
    def tag_regex(self) -> Pattern[str]:
        return self._tag_regex

    @tag_regex.setter
    def tag_regex(self, value: str | Pattern[str]) -> None:
        self._tag_regex = _check_tag_regex(value)

    @classmethod
    def from_file(
        cls,
        name: str = "pyproject.toml",
        dist_name: str | None = None,
        _load_toml: Callable[[str], dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Configuration:
        """
        Read Configuration from pyproject.toml (or similar).
        Raises exceptions when file is not found or toml is
        not installed or the file has invalid format or does
        not contain the [tool.setuptools_scm] section.
        """

        pyproject_data = _read_pyproject(name, _load_toml=_load_toml)
        args = _get_args_for_pyproject(pyproject_data, dist_name, kwargs)

        return cls(relative_to=name, **args)
