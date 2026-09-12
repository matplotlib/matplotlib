import asyncio
import os
import statistics
import sys
import time
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


@pytest.mark.parametrize("host, origin, allowed", [
    ("localhost:8988", "http://localhost:8988", True),
    ("localhost:8988", "http://evil.com", False),
    ("localhost:8988", "http://127.0.0.1:8988", False),
    ("localhost:8988", "http://[::1]:8988", False),
    ("127.0.0.1:8988", "http://127.0.0.1:8988", True),
    ("127.0.0.1:8988", "http://localhost:8988", False),
    ("127.0.0.1:8988", "http://[::1]:8988", False),
    ("[::1]:8988", "http://[::1]:8988", True),
    ("[::1]:8988", "http://[::2]:8988", False),
    ("[::1]:8988", "http://localhost:8988", False),
    ("[::1]:8988", "http://evil.com", False),
])
def test_websocket_rejects_cross_origin(host, origin, allowed):
    """Verify Tornado's default check_origin rejects cross-origin requests."""
    pytest.importorskip("tornado")
    from matplotlib.backends.backend_webagg import WebAggApplication

    ws = WebAggApplication.WebSocket.__new__(WebAggApplication.WebSocket)
    ws.request = MagicMock()
    ws.request.headers = {"Host": host}
    assert ws.check_origin(origin) is allowed


async def _run_asyncio_timer(single_shot, callback_s, loop_s):
    interval = 0.05
    timer = matplotlib.backends.backend_webagg_core.TimerAsyncio(interval * 1000)
    timer.single_shot = single_shot
    fires = []
    timer.add_callback(
        lambda: (fires.append(asyncio.get_running_loop().time()),
                 time.sleep(callback_s)))
    timer.start()
    await asyncio.sleep(loop_s)
    timer.stop()
    return interval, fires


def test_asyncio_timer_single_shot():
    _, fires = asyncio.run(_run_asyncio_timer(True, 0, 0.3))
    assert len(fires) == 1


def test_asyncio_timer_no_drift():
    # A slow callback must not push back the firings that follow it.
    interval, fires = asyncio.run(_run_asyncio_timer(False, 0.04, 0.5))
    assert len(fires) >= 4
    spacing = statistics.median(b - a for a, b in zip(fires, fires[1:]))
    assert spacing < interval * 1.4
