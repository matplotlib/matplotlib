from __future__ import annotations

import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import TYPE_CHECKING

from .setuptools import read_dist_name_from_setup_cfg

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

_ROOT = "root"
TOML_RESULT: TypeAlias = Dict[str, Any]
TOML_LOADER: TypeAlias = Callable[[str], TOML_RESULT]


class PyProjectData(NamedTuple):
    name: str
    tool_name: str
    project: TOML_RESULT
    section: TOML_RESULT

    @property
    def project_name(self) -> str | None:
        return self.project.get("name")


def lazy_tomli_load(data: str) -> TOML_RESULT:
    from tomli import loads

    return loads(data)


def read_pyproject(
    name: str = "pyproject.toml",
    tool_name: str = "setuptools_scm",
    _load_toml: TOML_LOADER | None = None,
) -> PyProjectData:
    if _load_toml is None:
        _load_toml = lazy_tomli_load
    with open(name, encoding="UTF-8") as strm:
        data = strm.read()
    defn = _load_toml(data)
    try:
        section = defn.get("tool", {})[tool_name]
    except LookupError as e:
        raise LookupError(f"{name} does not contain a tool.{tool_name} section") from e
    project = defn.get("project", {})
    return PyProjectData(name, tool_name, project, section)


def get_args_for_pyproject(
    pyproject: PyProjectData,
    dist_name: str | None,
    kwargs: TOML_RESULT,
) -> TOML_RESULT:
    """drops problematic details and figures the distribution name"""
    section = pyproject.section.copy()
    kwargs = kwargs.copy()
    if "relative_to" in section:
        relative = section.pop("relative_to")
        warnings.warn(
            f"{pyproject.name}: at [tool.{pyproject.tool_name}]\n"
            f"ignoring value relative_to={relative!r}"
            " as its always relative to the config file"
        )
    if "dist_name" in section:
        if dist_name is None:
            dist_name = section.pop("dist_name")
        else:
            assert dist_name == section["dist_name"]
            del section["dist_name"]
    if dist_name is None:
        # minimal pep 621 support for figuring the pretend keys
        dist_name = pyproject.project_name
    if dist_name is None:
        dist_name = read_dist_name_from_setup_cfg()
    if _ROOT in kwargs:
        if kwargs[_ROOT] is None:
            kwargs.pop(_ROOT, None)
        elif _ROOT in section:
            if section[_ROOT] != kwargs[_ROOT]:
                warnings.warn(
                    f"root {section[_ROOT]} is overridden"
                    f" by the cli arg {kwargs[_ROOT]}"
                )
            section.pop("root", None)
    return {"dist_name": dist_name, **section, **kwargs}
