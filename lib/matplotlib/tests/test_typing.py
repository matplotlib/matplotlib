import re
import typing
from pathlib import Path

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
