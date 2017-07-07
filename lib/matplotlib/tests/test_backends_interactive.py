import importlib
import os
import sys

from matplotlib.compat.subprocess import Popen
import pytest


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on Travis.  They are not available for all tested Python
# versions so we don't fail on missing backends.
#
# We also don't test on Py2 because its subprocess module doesn't support
# timeouts, and it would require a separate code path to check for module
# existence without actually trying to import the module (which may install
# an undesirable input hook).


def _get_testable_interactive_backends():
    return [
        pytest.mark.skipif(
            not os.environ.get("DISPLAY")
            or sys.version_info < (3,)
            or importlib.util.find_spec(module_name) is None,
            reason="No $DISPLAY or could not import {!r}".format(module_name))(
                backend)
        for module_name, backend in [
                ("PyQt5", "qt5agg"),
                ("tkinter", "tkagg"),
                ("wx", "wxagg")]]


_test_script = """\
import sys
from matplotlib import pyplot as plt

fig = plt.figure()
fig.canvas.mpl_connect("draw_event", lambda event: sys.exit())
plt.show()
"""


@pytest.mark.parametrize("backend", _get_testable_interactive_backends())
def test_backend(backend):
    environ = os.environ.copy()
    environ["MPLBACKEND"] = backend
    proc = Popen([sys.executable, "-c", _test_script], env=environ)
    # Empirically, 1s is not enough on Travis.
    assert proc.wait(timeout=5) == 0
