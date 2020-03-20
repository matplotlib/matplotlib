from matplotlib.tests.test_path import _test_curves

import numpy as np
import pytest


# all tests here are currently comparing against integrals
integrate = pytest.importorskip('scipy.integrate')


# get several curves to test our code on by borrowing the tests cases used in
# `~.tests.test_path`. get last path element ([-1]) and curve, not code ([0])
_test_curves = [list(tc.path.iter_bezier())[-1][0] for tc in _test_curves]


def _integral_arc_area(B):
    """(Signed) area swept out by ray from origin to curve."""
    dB = B.differentiate(B)
    def integrand(t):
        return np.cross(B(t), dB(t))/2
    return integrate.quad(integrand, 0, 1)[0]


@pytest.mark.parametrize("B", _test_curves)
def test_area_formula(B):
    assert np.isclose(_integral_arc_area(B), B.arc_area)
