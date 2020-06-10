import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

import matplotlib as mpl


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on CI.  They are not available for all tested Python
# versions so we don't fail on missing backends.

def _get_testable_interactive_backends():
    backends = []
    for deps, backend in [
            (["cairo", "gi"], "gtk3agg"),
            (["cairo", "gi"], "gtk3cairo"),
            (["PyQt5"], "qt5agg"),
            (["PyQt5", "cairocffi"], "qt5cairo"),
            (["PySide2"], "qt5agg"),
            (["PySide2", "cairocffi"], "qt5cairo"),
            (["tkinter"], "tkagg"),
            (["wx"], "wx"),
            (["wx"], "wxagg"),
            (["matplotlib.backends._macosx"], "macosx"),
    ]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if sys.platform == "linux" and not os.environ.get("DISPLAY"):
            reason = "$DISPLAY is unset"
        elif missing:
            reason = "{} cannot be imported".format(", ".join(missing))
        elif backend == 'macosx' and os.environ.get('TF_BUILD'):
            reason = "macosx backend fails on Azure"
        if reason:
            backend = pytest.param(
                backend,
                marks=pytest.mark.skip(
                    reason=f"Skipping {backend} because {reason}"))
        elif backend.startswith('wx') and sys.platform == 'darwin':
            # ignore on OSX because that's currently broken (github #16849)
            backend = pytest.param(
                backend,
                marks=pytest.mark.xfail(reason='github #16849'))
        backends.append(backend)
    return backends


# Using a timer not only allows testing of timers (on other backends), but is
# also necessary on gtk3 and wx, where a direct call to key_press_event("q")
# from draw_event causes breakage due to the canvas widget being deleted too
# early.  Also, gtk3 redefines key_press_event with a different signature, so
# we directly invoke it from the superclass instead.
_test_script = """\
import importlib
import importlib.util
import io
import json
import sys
from unittest import TestCase

import matplotlib as mpl
from matplotlib import pyplot as plt, rcParams
from matplotlib.backend_bases import FigureCanvasBase
rcParams.update({
    "webagg.open_in_browser": False,
    "webagg.port_retries": 1,
})
if len(sys.argv) >= 2:  # Second argument is json-encoded rcParams.
    rcParams.update(json.loads(sys.argv[1]))
backend = plt.rcParams["backend"].lower()
assert_equal = TestCase().assertEqual
assert_raises = TestCase().assertRaises

if backend.endswith("agg") and not backend.startswith(("gtk3", "web")):
    # Force interactive framework setup.
    plt.figure()

    # Check that we cannot switch to a backend using another interactive
    # framework, but can switch to a backend using cairo instead of agg, or a
    # non-interactive backend.  In the first case, we use tkagg as the "other"
    # interactive backend as it is (essentially) guaranteed to be present.
    # Moreover, don't test switching away from gtk3 (as Gtk.main_level() is
    # not set up at this point yet) and webagg (which uses no interactive
    # framework).

    if backend != "tkagg":
        with assert_raises(ImportError):
            mpl.use("tkagg", force=True)

    def check_alt_backend(alt_backend):
        mpl.use(alt_backend, force=True)
        fig = plt.figure()
        assert_equal(
            type(fig.canvas).__module__,
            "matplotlib.backends.backend_{}".format(alt_backend))

    if importlib.util.find_spec("cairocffi"):
        check_alt_backend(backend[:-3] + "cairo")
    check_alt_backend("svg")

mpl.use(backend, force=True)

fig, ax = plt.subplots()
assert_equal(
    type(fig.canvas).__module__,
    "matplotlib.backends.backend_{}".format(backend))

ax.plot([0, 1], [2, 3])

timer = fig.canvas.new_timer(1.)  # Test that floats are cast to int as needed.
timer.add_callback(FigureCanvasBase.key_press_event, fig.canvas, "q")
# Trigger quitting upon draw.
fig.canvas.mpl_connect("draw_event", lambda event: timer.start())
fig.canvas.mpl_connect("close_event", print)

result = io.BytesIO()
fig.savefig(result, format='png')

plt.show()

# Ensure that the window is really closed.
plt.pause(0.5)

# Test that saving works after interactive window is closed, but the figure is
# not deleted.
result_after = io.BytesIO()
fig.savefig(result_after, format='png')

if not backend.startswith('qt5') and sys.platform == 'darwin':
    # FIXME: This should be enabled everywhere once Qt5 is fixed on macOS to
    # not resize incorrectly.
    assert_equal(result.getvalue(), result_after.getvalue())
"""
_test_timeout = 10  # Empirically, 1s is not enough on Travis.


@pytest.mark.parametrize("backend", _get_testable_interactive_backends())
@pytest.mark.parametrize("toolbar", ["toolbar2", "toolmanager"])
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(backend, toolbar):
    if backend == "macosx" and toolbar == "toolmanager":
        pytest.skip("toolmanager is not implemented for macosx.")
    proc = subprocess.run(
        [sys.executable, "-c", _test_script,
         json.dumps({"toolbar": toolbar})],
        env={**os.environ, "MPLBACKEND": backend, "SOURCE_DATE_EPOCH": "0"},
        timeout=_test_timeout,
        stdout=subprocess.PIPE, universal_newlines=True)
    if proc.returncode:
        pytest.fail("The subprocess returned with non-zero exit status "
                    f"{proc.returncode}.")
    assert proc.stdout.count("CloseEvent") == 1


@pytest.mark.skipif('TF_BUILD' in os.environ,
                    reason="this test fails an azure for unknown reasons")
@pytest.mark.skipif(os.name == "nt", reason="Cannot send SIGINT on Windows.")
def test_webagg():
    pytest.importorskip("tornado")
    proc = subprocess.Popen([sys.executable, "-c", _test_script],
                            env={**os.environ, "MPLBACKEND": "webagg",
                                 "SOURCE_DATE_EPOCH": "0"})
    url = "http://{}:{}".format(
        mpl.rcParams["webagg.address"], mpl.rcParams["webagg.port"])
    timeout = time.perf_counter() + _test_timeout
    while True:
        try:
            retcode = proc.poll()
            # check that the subprocess for the server is not dead
            assert retcode is None
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
