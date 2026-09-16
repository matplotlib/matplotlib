import numpy as np
from numpy.testing import assert_array_equal
import matplotlib.pyplot as plt


def test_stem_remove():
    ax = plt.gca()
    st = ax.stem([1, 2], [1, 2])
    st.remove()


def test_errorbar_remove():

    # Regression test for a bug that caused remove to fail when using
    # fmt='none'

    ax = plt.gca()

    eb = ax.errorbar([1], [1])
    eb.remove()

    eb = ax.errorbar([1], [1], xerr=1)
    eb.remove()

    eb = ax.errorbar([1], [1], yerr=2)
    eb.remove()

    eb = ax.errorbar([1], [1], xerr=[2], yerr=2)
    eb.remove()

    eb = ax.errorbar([1], [1], fmt='none')
    eb.remove()


def test_nonstring_label():
    # Test for #26824
    plt.bar(np.arange(10), np.random.rand(10), label=1)
    plt.legend()


def test_barcontainer_position_centers__bottoms__tops():
    fig, ax = plt.subplots()
    pos = [1, 2, 4]
    bottoms = np.array([1, 5, 3])
    heights = np.array([2, 3, 4])

    container = ax.bar(pos, heights, bottom=bottoms)
    assert_array_equal(container.position_centers, pos)
    assert_array_equal(container.bottoms, bottoms)
    assert_array_equal(container.tops, bottoms + heights)

    container = ax.barh(pos, heights, left=bottoms)
    assert_array_equal(container.position_centers, pos)
    assert_array_equal(container.bottoms, bottoms)
    assert_array_equal(container.tops, bottoms + heights)
