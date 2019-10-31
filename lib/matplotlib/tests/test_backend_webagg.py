import subprocess
import os
import sys
import pytest


@pytest.mark.parametrize("backend", ["webagg", "nbagg"])
def test_webagg_fallback(backend):
    if backend == "nbagg":
        pytest.importorskip("IPython")
    env = {}
    if os.name == "nt":
        env = dict(os.environ)
    else:
        env = {"DISPLAY": ""}

    env["MPLBACKEND"] = backend

    test_code = (
        "import os;"
        + f"assert os.environ['MPLBACKEND'] == '{backend}';"
        + "import matplotlib.pyplot as plt; "
        + "print(plt.get_backend());"
        f"assert '{backend}' == plt.get_backend().lower();"
    )
    ret = subprocess.call([sys.executable, "-c", test_code], env=env)

    assert ret == 0
