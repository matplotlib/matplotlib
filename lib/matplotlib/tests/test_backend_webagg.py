import asyncio
import os
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


async def _check_completed_single_shot_property_update(timer_cls):
    timer = timer_cls(20)
    timer.single_shot = True
    fired = asyncio.Event()
    calls = 0

    def callback():
        nonlocal calls
        calls += 1
        fired.set()

    timer.add_callback(callback)
    timer.start()
    await asyncio.wait_for(fired.wait(), timeout=1)

    timer.interval = 30
    timer.single_shot = False
    await asyncio.sleep(0.1)
    assert calls == 1

    timer.single_shot = True
    fired.clear()
    timer.start()
    await asyncio.wait_for(fired.wait(), timeout=1)
    assert calls == 2


async def _check_single_shot_restart_from_callback(timer_cls):
    timer = timer_cls(20)
    timer.single_shot = True
    finished = asyncio.Event()
    calls = 0

    def callback():
        nonlocal calls
        calls += 1
        if calls == 1:
            timer.start()
        else:
            finished.set()

    timer.add_callback(callback)
    timer.start()
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert calls == 2


def test_asyncio_completed_single_shot_property_update_does_not_restart():
    asyncio.run(_check_completed_single_shot_property_update(
        matplotlib.backends.backend_webagg_core.TimerAsyncio))


def test_asyncio_single_shot_restart_from_callback():
    asyncio.run(_check_single_shot_restart_from_callback(
        matplotlib.backends.backend_webagg_core.TimerAsyncio))


def test_tornado_completed_single_shot_property_update_does_not_restart():
    tornado = pytest.importorskip("tornado")
    loop = tornado.ioloop.IOLoop()
    try:
        loop.run_sync(lambda: _check_completed_single_shot_property_update(
            matplotlib.backends.backend_webagg_core.TimerTornado))
    finally:
        loop.close()


def test_tornado_single_shot_restart_from_callback():
    tornado = pytest.importorskip("tornado")
    loop = tornado.ioloop.IOLoop()
    try:
        loop.run_sync(lambda: _check_single_shot_restart_from_callback(
            matplotlib.backends.backend_webagg_core.TimerTornado))
    finally:
        loop.close()


async def _time_n_fires(interval, callback_s, n, max_wait_s):
    # Collect n fire timestamps, bounded by max_wait_s in case a timer stalls.
    timer = matplotlib.backends.backend_webagg_core.TimerAsyncio(interval * 1000)
    fires = []
    done = asyncio.Event()

    def callback():
        fires.append(asyncio.get_running_loop().time())
        if callback_s:
            time.sleep(callback_s)
        if len(fires) >= n:
            done.set()

    timer.add_callback(callback)
    timer.start()
    try:
        await asyncio.wait_for(done.wait(), timeout=max_wait_s)
    except asyncio.TimeoutError:
        pass
    timer.stop()
    return fires


def test_asyncio_timer_no_drift():
    # A slow callback should skip to the next interval, not drift by
    # its own duration.  Each gap should land on a multiple of interval.
    interval = 1
    callback_s = interval / 2
    n = 4
    fires = asyncio.run(_time_n_fires(interval, callback_s, n, max_wait_s=20))
    assert len(fires) >= n, f"Only fired {len(fires)} times"
    for a, b in zip(fires, fires[1:]):
        gap = b - a
        offset = abs(gap - round(gap / interval) * interval)
        assert offset < interval * 0.3, (
            f"Gap {gap * 1000:.0f}ms is {offset * 1000:.0f}ms from the "
            f"nearest multiple of {interval * 1000:.0f}ms")
