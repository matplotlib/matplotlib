"""
===============================================
Let twin-axis aligned at the specified position
===============================================

Let the left-axis and right-axis aligned at the specified position.

In some data that need twin-axis plot, a point at left-axis and
another point at right-axis have same meaning, they shoud be aligned.
For example, we plot netvalue curve of a portfolio at twin-left-axis
and profit raito curve at twin-right-axis, the point 1.0 at the
left-axis and the point 0.0 at the right-axis both mean the begin
state of the portifolio, so they should be aligned.
"""

import matplotlib.pyplot as plt
import numpy as np

# Sim data
net_value = np.array([1.0, 0.98, 0.95, 1.0, 1.22, 1.25, 1.29, 1.34,
                      1.39, 1.18, 1.14, 1.04, 1.36, 1.41, 1.5, 1.56,
                      1.56, 1.57, 1.26, 1.35])
gain_pct = np.array([0.0, -0.02, -0.05, 0.0, 0.02, 0.04, 0.07,
                     0.12, 0.16, 0.18, 0.14, 0.04, 0.05, 0.08,
                     0.15, 0.2, 0.2, 0.21, 0.26, 0.35])


def twinxalign(ax_left, ax_right, v_left, v_right):
    """
    Let the position `v_left` on `ax_left`
    and the position `v_right` on `ax_right` be aligned
    """
    left_min, left_max = ax_left.get_ybound()
    right_min, right_max = ax_right.get_ybound()
    k = (left_max-left_min) / (right_max-right_min)
    b = left_min - k * right_min
    x_right_new = k * v_right + b
    dif = x_right_new - v_left
    if dif >= 0:
        right_min_new = ((left_min-dif) - b) / k
        k_new = (left_min-v_left) / (right_min_new-v_right)
        b_new = v_left - k_new * v_right
        right_max_new = (left_max - b_new) / k_new
    else:
        right_max_new = ((left_max-dif) - b) / k
        k_new = (left_max-v_left) / (right_max_new-v_right)
        b_new = v_left - k_new * v_right
        right_min_new = (left_min - b_new) / k_new
    ax_right.set_ylim([right_min_new, right_max_new])
    return ax_left, ax_right


def plot_example(align=False):
    """Plot sim data"""
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(111)
    ax1.plot(net_value, "-k")
    ax1.axhline(1, c="k", lw=1, ls="--")
    ax1.set_ylabel("NetValue(black)", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(gain_pct, "-b")
    ax2.axhline(0, c="r", lw=1, ls="--")
    ax2.set_ylabel("GainPct(blue)", fontsize=16)
    if align:
        twinxalign(ax1, ax2, 1, 0)
        plt.title("aligned", fontsize=16)
    else:
        plt.title("not align", fontsize=16)
    plt.show()


plot_example()
plot_example(align=True)
