import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_grouped_bar_single_hatch_str():
    fig, ax = plt.subplots()
    x = np.arange(3)
    heights = [np.array([1, 2, 3]), np.array([2, 1, 2])]
    containers = ax.grouped_bar(heights, positions=x, hatch='//')
    for c in containers.bar_containers:
        for rect in c:
            assert rect.get_hatch() == '//'


def test_grouped_bar_hatch_sequence():
    fig, ax = plt.subplots()
    x = np.arange(2)
    heights = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    hatches = ['//', 'xx', '..']
    containers = ax.grouped_bar(heights, positions=x, hatch=hatches)
    for gi, c in enumerate(containers.bar_containers):
        for rect in c:
            assert rect.get_hatch() == hatches[gi]


def test_grouped_bar_hatch_length_mismatch():
    fig, ax = plt.subplots()
    x = np.arange(2)
    heights = [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])]
    with pytest.raises(ValueError, match="hatch.*length"):
        ax.grouped_bar(heights, positions=x, hatch=['//'])
