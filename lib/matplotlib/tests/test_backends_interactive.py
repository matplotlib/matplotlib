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


_test_script = """\
import sys
from matplotlib import pyplot as plt, rcParams
rcParams.update({
    "webagg.open_in_browser": False,
    "webagg.port_retries": 1,
})

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2], [2, 3])
fig.canvas.mpl_connect("draw_event", lambda event: sys.exit())
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
