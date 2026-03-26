from matplotlib.ticker import WilkinsonLocator, MaxNLocator


def test_wilkinson_basic():
    loc = WilkinsonLocator(nbins=5)
    ticks = loc.tick_values(3, 97)

    assert len(ticks) >= 2
    assert ticks[0] <= 3
    assert ticks[-1] >= 97


def test_wilkinson_vs_maxn():
    w = WilkinsonLocator(nbins=5).tick_values(3, 97)
    m = MaxNLocator(nbins=5).tick_values(3, 97)

    # Wilkinson should not produce more ticks
    assert len(w) <= len(m)
