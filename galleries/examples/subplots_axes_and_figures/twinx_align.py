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
import pandas as pd
from matplotlib import pyplot as plt


# Sim data
net_in = [100, 0, 0, 0, 20, 0, 0, 0, 0, -20, 
          0, 0, 30, 0, 0, 0, 0, 0, -30, 0]
net_gain= [0, -2, -3, 5, 2, 3, 4, 5, 5, -1,
           -4, -10, 2, 5, 9, 6, 0, 1, -1, 9]
df = pd.DataFrame({"net_in": net_in, "net_gain": net_gain})
df["total_in"] = df["net_in"].cumsum()
df["value"] = df["total_in"] + df["net_gain"].cumsum()
df["value_net"] = df["value"] / df["value"].iloc[0] 
df["gain_pct"] = df["value"] / df["total_in"] - 1 


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
    def _forward(x):
        return k_new * x + b_new
    def _inverse(x):
        return (x - b_new) / k_new
    ax_right.set_ylim([right_min_new, right_max_new])
    ax_right.set_yscale("function", functions=(_forward, _inverse))
    return ax_left, ax_right


def general_plot():
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(111)
    ax1.plot(df["value_net"], "-k")
    ax1.axhline(1, c="k", lw=1, ls="--")
    ax1.set_ylabel("NetValue(black)", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(df["gain_pct"], "-b")
    ax2.axhline(0, c="r", lw=1, ls="--")
    ax2.set_ylabel("GainPct(blue)", fontsize=16)
    plt.title("no align plot", fontsize=16)
    plt.show()


def twinx_align_plot():
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(111)
    ax1.plot(df["value_net"], "-k")
    ax1.axhline(1, c="k", lw=1, ls="--")
    ax1.set_ylabel("NetValue(black)", fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(df["gain_pct"], "-b")
    ax2.axhline(0, c="r", lw=1, ls="--")
    ax2.set_ylabel("GainPct(blue)", fontsize=16)
    twinxalign(ax1, ax2, 1, 0)
    plt.title("align left-axis 1 with right-axis 0", fontsize=16)
    plt.show()


general_plot()
twinx_align_plot()




