import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


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
