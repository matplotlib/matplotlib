from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg") # non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.0

fig = plt.figure(figsize=(3, 4.5), dpi=150)

ax_big = fig.add_axes([0.05, 0.55, 0.9, 0.4]) 
ax_big.set_xlim(0, 6)
ax_big.set_ylim(0, 3)

big_ellipse = Ellipse(
    (3.5, 1.5),
    width=7.0,
    height=2.4,
    angle=30,
    facecolor="#ffffcc",
    edgecolor="black",
    linewidth=1.5,
)
ax_big.add_patch(big_ellipse)
ax_big.set_axis_off()

ax_small = fig.add_axes([0.35, 0.10, 0.30, 0.30])
ax_small.set_xlim(0, 3)
ax_small.set_ylim(0, 3)

small_ellipse = Ellipse(
    (1.5, 1.5),
    width=2.0,
    height=3.0,
    angle=30,
    facecolor="#ccffcc",
    edgecolor="black",
    linewidth=1.5,
)
ax_small.add_patch(small_ellipse)
ax_small.set_axis_off()

outfile = OUT_DIR / "ellipse_tgrohens_thr0.0.png"
fig.savefig(outfile, dpi=150)
plt.close(fig)

print("Salvo em:", outfile)
