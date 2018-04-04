import importlib
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

import matplotlib as mpl


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on Travis.  They are not available for all tested Python
# versions so we don't fail on missing backends.

def _get_testable_interactive_backends():
    backends = []
    for deps, backend in [(["cairocffi", "pgi"], "gtk3agg"),
                          (["cairocffi", "pgi"], "gtk3cairo"),
                          (["PyQt5"], "qt5agg"),
                          (["cairocffi", "PyQt5"], "qt5cairo"),
                          (["tkinter"], "tkagg"),
                          (["wx"], "wx"),
                          (["wx"], "wxagg")]:
        reason = None
        if not os.environ.get("DISPLAY"):
            reason = "No $DISPLAY"
        elif any(importlib.util.find_spec(dep) is None for dep in deps):
            reason = "Missing dependency"
        backends.append(pytest.mark.skip(reason=reason)(backend) if reason
                        else backend)
    return backends


# 1. Using a timer not only allows testing of timers (on other backends), but
#    is also necessary on gtk3 and wx, where a direct call to
#    key_press_event("q") from draw_event causes breakage due to the canvas
#    widget being deleted too early.
# 2. On gtk3, we cannot even test the timer setup (on Travis, which uses pgi)
#    due to https://github.com/pygobject/pgi/issues/45.  So we just cleanly
#    exit from the draw_event.
_test_script = """\
import sys
from matplotlib import pyplot as plt, rcParams
rcParams.update({
    "webagg.open_in_browser": False,
    "webagg.port_retries": 1,
})

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 1], [2, 3])

if rcParams["backend"].startswith("GTK3"):
    fig.canvas.mpl_connect("draw_event", lambda event: sys.exit(0))
else:
    timer = fig.canvas.new_timer(1)
    timer.add_callback(fig.canvas.key_press_event, "q")
    # Trigger quitting upon draw.
    fig.canvas.mpl_connect("draw_event", lambda event: timer.start())

plt.show()
"""
_test_timeout = 10  # Empirically, 1s is not enough on Travis.


@pytest.mark.parametrize("backend", _get_testable_interactive_backends())
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(backend):
    subprocess.run([sys.executable, "-c", _test_script],
                   env={**os.environ, "MPLBACKEND": backend},
                   check=True,  # Throw on failure.
                   timeout=_test_timeout)


@pytest.mark.skipif(os.name == "nt", reason="Cannot send SIGINT on Windows.")
def test_webagg():
    pytest.importorskip("tornado")
    proc = subprocess.Popen([sys.executable, "-c", _test_script],
                            env={**os.environ, "MPLBACKEND": "webagg"})
    url = "http://{}:{}".format(
        mpl.rcParams["webagg.address"], mpl.rcParams["webagg.port"])
    timeout = time.perf_counter() + _test_timeout
    while True:
        try:
            conn = urllib.request.urlopen(url)
            break
        except urllib.error.URLError:
            if time.perf_counter() > timeout:
                pytest.fail("Failed to connect to the webagg server.")
            else:
                continue
    conn.close()
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=_test_timeout) == 0
