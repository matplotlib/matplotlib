import os
import sys
from unittest.mock import MagicMock

import pytest

import matplotlib.backends.backend_webagg_core
from matplotlib.backends.backend_webagg_core import (
    FigureCanvasWebAggCore, NavigationToolbar2WebAgg,
)
from matplotlib.testing import subprocess_run_for_testing


@pytest.mark.parametrize("backend", ["webagg", "nbagg"])
def test_webagg_fallback(backend):
    pytest.importorskip("tornado")
    if backend == "nbagg":
        pytest.importorskip("IPython")
    env = dict(os.environ)
    if sys.platform != "win32":
        env["DISPLAY"] = ""

    env["MPLBACKEND"] = backend

    test_code = (
        "import os;"
        + f"assert os.environ['MPLBACKEND'] == '{backend}';"
        + "import matplotlib.pyplot as plt; "
        + "print(plt.get_backend());"
        f"assert '{backend}' == plt.get_backend().lower();"
    )
    subprocess_run_for_testing([sys.executable, "-c", test_code], env=env, check=True)


def test_webagg_core_no_toolbar():
    fm = matplotlib.backends.backend_webagg_core.FigureManagerWebAgg
    assert fm._toolbar2_class is None


def test_toolbar_button_dispatch_allowlist():
    """Only declared toolbar items should be dispatched."""
    fig = MagicMock()
    canvas = FigureCanvasWebAggCore(fig)
    canvas.toolbar = MagicMock(spec=NavigationToolbar2WebAgg)
    canvas.toolbar.toolitems = NavigationToolbar2WebAgg.toolitems

    # Valid toolbar action should be dispatched.
    canvas.handle_toolbar_button({'name': 'home'})
    canvas.toolbar.home.assert_called_once()

    # Invalid names should be silently ignored.
    canvas.toolbar.reset_mock()
    canvas.handle_toolbar_button({'name': '__init__'})
    canvas.handle_toolbar_button({'name': 'not_a_real_button'})
    # No methods should have been called.
    assert canvas.toolbar.method_calls == []


@pytest.mark.parametrize("host, origin", [
    ("localhost:8988", "http://example.com"),
    ("localhost:8988", "http://127.0.0.1:8988"),
    ("localhost:8988", "http://[::1]:8988"),
    ("127.0.0.1:8988", "http://localhost:8988"),
    ("127.0.0.1:8988", "http://[::1]:8988"),
    ("[::1]:8988", "http://[::2]:8988"),
    ("[::1]:8988", "http://localhost:8988"),
    ("[::1]:8988", "http://example.com"),
])
def test_websocket_rejects_cross_origin(host, origin):
    """Verify Tornado's default check_origin rejects cross-origin requests."""
    pytest.importorskip("tornado")
    from matplotlib.backends.backend_webagg import WebAggApplication

    app = WebAggApplication()
    request = MagicMock()
    request.headers = {"Host": host}
    ws = WebAggApplication.WebSocket(app, request)
    assert not ws.check_origin(origin)  # Not allowed.
