"""
Tests specific to the bezier module.
"""

from matplotlib.bezier import inside_circle, split_bezier_intersecting_with_closedpath
from matplotlib.tests.test_path import _test_curves

import numpy as np
import pytest


def test_split_bezier_with_large_values():
    # These numbers come from gh-27753
    arrow_path = [(96950809781500.0, 804.7503795623779),
                  (96950809781500.0, 859.6242585800646),
                  (96950809781500.0, 914.4981375977513)]
    in_f = inside_circle(96950809781500.0, 804.7503795623779, 0.06)
    split_bezier_intersecting_with_closedpath(arrow_path, in_f)
    # All we are testing is that this completes
    # The failure case is an infinite loop resulting from floating point precision
    # pytest will timeout if that occurs


# get several curves to test our code on by borrowing the tests cases used in
# `~.tests.test_path`. get last path element ([-1]) and curve, not code ([0])
_test_curves = [list(tc.path.iter_bezier())[-1][0] for tc in _test_curves]
# np2+ uses trapezoid, but we need to fallback to trapz for np<2 since it isn't there
_trapezoid = getattr(np, "trapezoid", np.trapz)  # type: ignore[attr-defined]


def _integral_arc_area(B):
    """(Signed) area swept out by ray from origin to curve."""
    dB = B.differentiate(B)
    def integrand(t):
        x = B(t)
        y = dB(t)
        # np.cross for 2d input
        return (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]) / 2
    x = np.linspace(0, 1, 1000)
    y = integrand(x)
    return _trapezoid(y, x)


@pytest.mark.parametrize("B", _test_curves)
def test_area_formula(B):
    assert np.isclose(_integral_arc_area(B), B.arc_area)
