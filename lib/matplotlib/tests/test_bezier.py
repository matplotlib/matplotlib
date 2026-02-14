"""
Tests specific to the bezier module.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from matplotlib.bezier import (
    _real_roots_in_01, inside_circle, split_bezier_intersecting_with_closedpath
)


def _np_real_roots_in_01(coeffs):
    """Reference implementation using np.roots for comparison."""
    coeffs = np.asarray(coeffs)
    # np.roots expects descending order (highest power first)
    all_roots = np.roots(coeffs[::-1])
    # Filter to real roots in [0, 1]
    real_mask = np.abs(all_roots.imag) < 1e-10
    real_roots = all_roots[real_mask].real
    in_range = (real_roots >= -1e-10) & (real_roots <= 1 + 1e-10)
    return np.sort(np.clip(real_roots[in_range], 0, 1))


@pytest.mark.parametrize("coeffs, expected", [
    ([-0.5, 1], [0.5]),
    ([-2, 1], []),                    # roots: [2.0], not in [0, 1]
    ([0.1875, -1, 1], [0.25, 0.75]),
    ([1, -2.5, 1], [0.5]),            # roots: [0.5, 2.0], only one in [0, 1]
    ([1, 0, 1], []),                  # roots: [+-i], not real
    ([-0.08, 0.66, -1.5, 1], [0.2, 0.5, 0.8]),
    ([5], []),
    ([0, 0, 0], []),
    ([0, -0.5, 1], [0.0, 0.5]),
    ([0.5, -1.5, 1], [0.5, 1.0]),
])
def test_real_roots_in_01_known_cases(coeffs, expected):
    """Test _real_roots_in_01 against known values and np.roots reference."""
    result = _real_roots_in_01(coeffs)
    np_expected = _np_real_roots_in_01(coeffs)
    assert_allclose(result, expected, atol=1e-10)
    assert_allclose(result, np_expected, atol=1e-10)


@pytest.mark.parametrize("degree", range(1, 11))
def test_real_roots_in_01_random(degree):
    """Test random polynomials against np.roots."""
    rng = np.random.default_rng(seed=0)
    coeffs = rng.uniform(-10, 10, size=degree + 1)
    result = _real_roots_in_01(coeffs)
    expected = _np_real_roots_in_01(coeffs)
    assert len(result) == len(expected)
    if len(result) > 0:
        assert_allclose(result, expected, atol=1e-8)


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
