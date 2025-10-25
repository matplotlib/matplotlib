import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_grouped_bar_single_hatch_str():
    """All bars should share the same hatch when a single string is passed."""
    fig, ax = plt.subplots()
    x = np.arange(3)
    heights = [np.array([1, 2, 3]), np.array([2, 1, 2])]
    containers = ax.grouped_bar(heights, positions=x, hatch='//')

    # Verify each bar has the same hatch pattern
    for c in containers.bar_containers:
        for rect in c:
            assert rect.get_hatch() == '//'


def test_grouped_bar_hatch_sequence():
    """Each dataset should receive its own hatch pattern when a sequence is passed."""
    fig, ax = plt.subplots()
    x = np.arange(2)
    heights = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    hatches = ['//', 'xx', '..']
    containers = ax.grouped_bar(heights, positions=x, hatch=hatches)

    # Verify each dataset gets the corresponding hatch
    for gi, c in enumerate(containers.bar_containers):
        for rect in c:
            assert rect.get_hatch() == hatches[gi]


def test_grouped_bar_hatch_length_mismatch():
    """Passing a hatch sequence with length different from
    number of datasets should raise an error.
    """

    fig, ax = plt.subplots()
    x = np.arange(2)
    heights = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    hatches = ['//', 'xx']  # only 2 hatches for 3 datasets
    with pytest.raises(ValueError, match="Expected 3 hatches, got 2"):
        ax.grouped_bar(heights, positions=x, hatch=hatches)


def test_grouped_bar_hatch_none():
    """Passing hatch=None should result in bars with no hatch."""
    fig, ax = plt.subplots()
    x = np.arange(2)
    heights = [np.array([1, 2]), np.array([2, 3])]
    containers = ax.grouped_bar(heights, positions=x, hatch=None)

    for c in containers.bar_containers:
        for rect in c:
            assert rect.get_hatch() is None


def test_grouped_bar_hatch_mixed_orientation():
    """Ensure hatch works correctly for both vertical and horizontal orientations."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x = np.arange(3)
    heights = [np.array([1, 2, 3]), np.array([2, 1, 2])]
    hatches = ['//', 'xx']

    containers_v = ax1.grouped_bar(
        heights, positions=x, hatch=hatches, orientation="vertical")
    containers_h = ax2.grouped_bar(
        heights, positions=x, hatch=hatches, orientation="horizontal")

    for gi, (cv, ch) in enumerate(
            zip(containers_v.bar_containers, containers_h.bar_containers)):
        for rect in cv:
            assert rect.get_hatch() == hatches[gi]
        for rect in ch:
            assert rect.get_hatch() == hatches[gi]
