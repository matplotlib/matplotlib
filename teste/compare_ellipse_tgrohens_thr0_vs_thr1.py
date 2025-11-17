# teste/compare_ellipse_tgrohens_thr0_vs_thr1.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

img0 = plt.imread(BASE / "ellipse_tgrohens_thr0.0.png")
img1 = plt.imread(BASE / "ellipse_tgrohens_thr1.0.png")

h = min(img0.shape[0], img1.shape[0])
w = min(img0.shape[1], img1.shape[1])

img0 = img0[:h, :w, :3]
img1 = img1[:h, :w, :3]

# 2) Overlay R/G (origin on channel R, patch on channel G)
overlay = np.zeros((h, w, 3), dtype=img0.dtype)
overlay[..., 0] = img0[..., 0]  # R thr=0.0
overlay[..., 1] = img1[..., 1]  # G thr=1.0
# B = 0

# 3) Abs diff by pixel
diff = np.abs(img0.astype(float) - img1.astype(float)).max(axis=2)

# 4) Fig 2x2: original, patch, overlay, diff
fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
ax00, ax01, ax10, ax11 = axes.ravel()

ax00.imshow(img0)
ax00.set_title("ellipse thr=0.0")
ax00.axis("off")

ax01.imshow(img1)
ax01.set_title("ellipse thr=1.0")
ax01.axis("off")

ax10.imshow(overlay)
ax10.set_title("overlay (orig=R, patch=G)")
ax10.axis("off")

im = ax11.imshow(diff, cmap="magma", origin="upper")
ax11.set_title("abs diff")
ax11.axis("off")
fig.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(BASE / "compare_ellipse_tgrohens_thr0_vs_thr1.png", dpi=150)
plt.close(fig)