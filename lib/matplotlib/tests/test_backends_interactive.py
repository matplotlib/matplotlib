import importlib
import os
import sys

from matplotlib.compat.subprocess import Popen
import pytest


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on Travis.  They are not available for all tested Python
# versions so we don't fail on missing backends.
#
# Once the Travis build environment switches to Ubuntu 14.04, we should be able
# to add wxagg (which has wheels for 14.04 but not for 12.04).
#
# We also don't test on Py2 because its subprocess module doesn't support
# timeouts, and it would require a separate code path to check for module
# existence without actually trying to import the module (which may install
# an undesirable input hook).


def _get_available_backends():
    if sys.version_info < (3,):
        return []
    else:
        return [
            pytest.mark.skipif(
                importlib.util.find_spec(module_name) is None,
                reason="Could not import {!r}".format(module_name))(backend)
            for module_name, backend in [
                ("PyQt5", "qt5agg"),
                ("tkinter", "tkagg")]]


_test_script = """\
import sys
from matplotlib import pyplot as plt

fig = plt.figure()
fig.canvas.mpl_connect("draw_event", lambda event: sys.exit())
plt.show()
"""


@pytest.mark.skipif("DISPLAY" not in os.environ,
                    reason="The DISPLAY environment variable is not set.")
@pytest.mark.parametrize("backend", _get_available_backends())
def test_backend(backend):
    environ = os.environ.copy()
    environ["MPLBACKEND"] = backend
    proc = Popen([sys.executable, "-c", _test_script], env=environ)
    # Empirically, 1s is not enough on Travis.
    assert proc.wait(timeout=5) == 0
