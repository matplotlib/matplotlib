import ast
import re
import typing
from pathlib import Path

import pytest

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.typing import RcKeyType, RcGroupKeyType


def test_cm_stub_matches_runtime_colormaps():
    runtime_cm = plt.cm
    runtime_cmaps = {
        name
        for name, value in vars(runtime_cm).items()
        if isinstance(value, Colormap)
    }

    cm_pyi_path = Path(__file__).parent.parent / "cm.pyi"
    assert cm_pyi_path.exists(), f"{cm_pyi_path} does not exist"

    pyi_content = cm_pyi_path.read_text(encoding='utf-8')

    stubbed_cmaps = set(
        re.findall(r"^(\w+):\s+colors\.Colormap", pyi_content, re.MULTILINE)
    )

    assert runtime_cmaps, (
        "No colormaps variables found at runtime in matplotlib.colors"
    )
    assert stubbed_cmaps, (
        "No colormaps found in cm.pyi"
    )

    assert runtime_cmaps == stubbed_cmaps


def test_typing_aliases_documented():
    """Every public type in typing.py is documented or intentionally undocumented."""
    typing_docs = Path(__file__).parents[3] / "doc/api/typing_api.rst"
    if not typing_docs.exists():
        pytest.skip("Documentation sources not available")

    typing_py_path = Path(__file__).parents[1] / "typing.py"
    assert typing_py_path.exists(), f"{typing_py_path} does not exist"
    tree = ast.parse(typing_py_path.read_text(encoding="utf-8"))

    # Collect all public module-level assignment names (both annotated and plain).
    defined_types = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else None
        else:
            continue
        if name is not None and not name.startswith("_"):
            defined_types.add(name)

    assert defined_types, "No type definitions found in typing.py"

    # Collect documented ``.. autodata::`` entries.
    rst_content = typing_docs.read_text(encoding="utf-8")
    documented = set(
        re.findall(
            r"^\.\.\s+autodata::\s+matplotlib\.typing\.(\w+)",
            rst_content,
            re.MULTILINE,
        )
    )
    assert documented, "No autodata entries found in typing_api.rst"

    # Collect types listed under the comment
    # ".. intentionally undocumented types (one type per row)".
    # Each type must be indented and an empty line ends the section.
    intentionally_undocumented = set()
    marker = ".. intentionally undocumented types (one type per row)"
    lines_following_marker = rst_content.split(marker, 1)[1].splitlines()[1:]
    for line in lines_following_marker:
        if not line or not line[0].isspace():
            break
        intentionally_undocumented.add(line.strip())

    accounted_for = documented | intentionally_undocumented

    missing = defined_types - accounted_for
    assert not missing, (
        f"Types defined in typing.py but not in typing_api.rst "
        f"(document them or add to 'intentionally undocumented types'): {missing}"
    )

    extra = accounted_for - defined_types
    assert not extra, (
        f"Types listed in typing_api.rst but not defined in typing.py: {extra}"
    )


def test_rcparam_stubs():
    runtime_rc_keys = {
        name for name in plt.rcParamsDefault.keys()
        if not name.startswith('_')
    }

    assert {*typing.get_args(RcKeyType)} == runtime_rc_keys

    runtime_rc_group_keys = set()
    for name in runtime_rc_keys:
        groups = name.split('.')
        for i in range(1, len(groups)):
            runtime_rc_group_keys.add('.'.join(groups[:i]))

    assert {*typing.get_args(RcGroupKeyType)} == runtime_rc_group_keys
