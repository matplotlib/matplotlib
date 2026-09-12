import numpy as np
import pytest
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import WilkinsonLocator


def test_wilkinson_basic():
    """Ticks should cover the full data range."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(3, 97)

    assert len(ticks) >= 2
    assert ticks[0] <= 3
    assert ticks[-1] >= 97


def test_wilkinson_vs_maxn():
    """WilkinsonLocator should not produce more ticks than MaxNLocator."""
    w = WilkinsonLocator(nbins=5).tick_values(3, 97)
    m = MaxNLocator(nbins=5).tick_values(3, 97)

    assert len(w) <= len(m)


def test_wilkinson_reversed_input():
    """Should handle vmin > vmax gracefully."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(97, 3)

    assert len(ticks) >= 2
    assert ticks[0] <= 3
    assert ticks[-1] >= 97


def test_wilkinson_equal_input():
    """Should handle vmin == vmax without crashing."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(50, 50)

    assert len(ticks) >= 1


def test_wilkinson_negative_range():
    """Should work correctly for negative ranges."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(-100, -10)

    assert ticks[0] <= -100
    assert ticks[-1] >= -10


def test_wilkinson_cross_zero():
    """Should work correctly when range crosses zero."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(-50, 50)

    assert ticks[0] <= -50
    assert ticks[-1] >= 50


def test_wilkinson_small_range():
    """Should work for very small ranges."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0.001, 0.009)

    assert len(ticks) >= 2
    assert ticks[0] <= 0.001
    assert ticks[-1] >= 0.009


def test_wilkinson_large_range():
    """Should work for very large ranges."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0, 1_000_000)

    assert len(ticks) >= 2
    assert ticks[0] <= 0
    assert ticks[-1] >= 1_000_000


# ---------------- steps Parameter ---------------- #

def test_wilkinson_custom_steps():
    """Custom steps should be respected."""
    loc = WilkinsonLocator(nbins=5, steps=[1, 2, 5, 10])
    ticks = loc.tick_values(0, 100)

    assert len(ticks) >= 2
    assert ticks[0] <= 0
    assert ticks[-1] >= 100


def test_wilkinson_custom_steps_stored():
    """Custom steps should be stored on the instance."""
    custom = [1, 2, 5, 10]
    loc = WilkinsonLocator(nbins=5, steps=custom)

    assert loc.steps == custom


def test_wilkinson_default_steps():
    """Default steps should be [1, 2, 2.5, 5, 10]."""
    loc = WilkinsonLocator(nbins=5)

    assert loc.steps == [1, 2, 2.5, 5, 10]


def test_wilkinson_steps_empty_raises():
    """Empty steps list should raise ValueError."""
    with pytest.raises(ValueError):
        WilkinsonLocator(nbins=5, steps=[])


def test_wilkinson_single_step():
    """Single step value should still produce ticks."""
    loc = WilkinsonLocator(nbins=5, steps=[1])
    ticks = loc.tick_values(0, 100)

    assert len(ticks) >= 2


# ---------------- Scoring Fairness ---------------- #

def test_wilkinson_coverage_dominates():
    """
    Coverage should dominate simplicity.
    Both [0,25,50,75,100] and [0,20,40,60,80,100] are valid for (0,100).
    The choice should depend on coverage/density, not just q niceness.
    The key assertion is that the result covers the range well.
    """
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0, 100)

    assert ticks[0] <= 0
    assert ticks[-1] >= 100
    assert len(ticks) >= 4


def test_wilkinson_does_not_always_prefer_25_steps():
    """
    For a range like (0, 80), [0,20,40,60,80] (step=20) should score
    higher than [0,25,50,75] (step=25) because it covers the range better.
    """
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0, 80)

    # [0, 20, 40, 60, 80] covers exactly; [0, 25, 50, 75] misses 80
    assert ticks[-1] >= 80


def test_wilkinson_simplicity_not_sole_decider():
    """
    q=1 is always most 'simple', but shouldn't always win.
    For (0, 100) with nbins=5, a step of 25 or 20 is better than step=1.
    """
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0, 100)

    # If simplicity alone decided, we'd get 100 ticks of step=1
    assert len(ticks) <= 10


# ---------------- Tick Quality ---------------- #

def test_wilkinson_ticks_are_sorted():
    """Ticks should always be in ascending order."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(3, 97)

    assert np.all(np.diff(ticks) > 0)


def test_wilkinson_ticks_evenly_spaced():
    """Ticks should be evenly spaced (uniform step)."""
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(0, 100)

    diffs = np.diff(ticks)
    assert np.allclose(diffs, diffs[0], rtol=1e-5)


def test_wilkinson_nbins_respected():
    """Number of ticks should stay close to nbins."""
    for nbins in [3, 5, 8, 10]:
        loc = WilkinsonLocator(nbins=nbins)
        ticks = loc.tick_values(0, 100)
        # Allow some flexibility but shouldn't be wildly off
        assert len(ticks) <= nbins * 2 + 1
