import os
import subprocess
import tempfile
from PIL import Image

import pytest

import matplotlib
from matplotlib.backend_tools import ToolHelpBase, ToolToggleBase
import matplotlib.pyplot as plt
from matplotlib.testing import subprocess_run_helper
from .test_backends_interactive import (
        _get_testable_interactive_backends, _test_timeout,
        )


@pytest.mark.parametrize('rc_shortcut,expected', [
    ('home', 'Home'),
    ('backspace', 'Backspace'),
    ('f1', 'F1'),
    ('ctrl+a', 'Ctrl+A'),
    ('ctrl+A', 'Ctrl+Shift+A'),
    ('a', 'a'),
    ('A', 'A'),
    ('ctrl+shift+f1', 'Ctrl+Shift+F1'),
    ('1', '1'),
    ('cmd+p', 'Cmd+P'),
    ('cmd+1', 'Cmd+1'),
])
def test_format_shortcut(rc_shortcut, expected):
    assert ToolHelpBase.format_shortcut(rc_shortcut) == expected


def _test_toolbar_button_la_mode_icon_inside_subprocess():
    matplotlib.rcParams["toolbar"] = "toolmanager"
    # create an icon in LA mode
    with tempfile.TemporaryDirectory() as tempdir:
        img = Image.new("LA", (26, 26))
        tmp_img_path = os.path.join(tempdir, "test_la_icon.png")
        img.save(tmp_img_path)

        class CustomTool(ToolToggleBase):
            image = tmp_img_path
            description = ""  # gtk3 backend does not allow None

        fig = plt.figure()
        toolmanager = fig.canvas.manager.toolmanager
        toolbar = fig.canvas.manager.toolbar
        toolmanager.add_tool("test", CustomTool)
        toolbar.add_tool("test", "group")


@pytest.mark.parametrize(
        "env",
        _get_testable_interactive_backends(),
        )
def test_toolbar_button_la_mode_icon(env):
    # test that icon in LA mode can be used for buttons
    # see GH#25164
    try:
        # run inside subprocess for a self-contained environment
        proc = subprocess_run_helper(
            _test_toolbar_button_la_mode_icon_inside_subprocess,
            timeout=_test_timeout,
            extra_env=env,
            )
    except subprocess.CalledProcessError as err:
        pytest.fail(
                f"subprocess failed to test intended behavior: {err.stderr}")
