import subprocess
import sys
import pytest


@pytest.mark.parametrize('backend', ['webagg', 'nbagg'])
def test_webagg_fallback(backend):
    test_code = ("import matplotlib.pyplot as plt; " +
                 "print(plt.get_backend());"
                 f"assert '{backend}' == plt.get_backend().lower();")
    ret = subprocess.call(
        [sys.executable, "-c", test_code],
        env={"MPLBACKEND": backend, "DISPLAY": ""}
    )

    assert ret == 0
