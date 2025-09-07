"""
==========================================
Alpha parameter behavior with different image types
==========================================

Demonstrate how the alpha parameter interacts with different image data types
(2D arrays, RGB, RGBA) and colormaps in matplotlib's imshow function.

This example shows the behavior of the alpha parameter when applied to:
- 2D scalar data with default colormap
- 2D scalar data with custom alpha-aware colormap
- RGB images
- RGBA images with existing alpha channels

The alpha parameter can be:
- None (default, no transparency)
- A scalar float (uniform transparency)
- A 2D array (per-pixel transparency)
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, axs = plt.subplots(3, 4, figsize=(12, 10), layout="compressed")

# Set red background to make transparency visible
for ax in axs.flat:
    ax.set(facecolor="red", xticks=[], yticks=[])

# Create test data
mapped = np.array([[0.1, 1.0], [1.0, 0.1]])
rgb = np.repeat(mapped[:, :, np.newaxis], 3, axis=2)
rgba = np.concatenate(
    [
        rgb,
        [
            [[1.0], [0.9]],
            [[0.8], [0.7]],
        ],
    ],
    axis=2,
)

alpha_scalar = 0.5
alpha_2d = np.full_like(mapped, alpha_scalar)

# Create a colormap with built-in alpha
cmap_with_alpha = ListedColormap(
    np.concatenate(
        [plt.cm.viridis.colors, np.full((len(plt.cm.viridis.colors), 1), alpha_scalar)],
        axis=1,
    ),
)

# Test different alpha parameter combinations
for ax, alpha, alpha_type in zip(axs, [None, alpha_scalar, alpha_2d],
                                ["None", "scalar", "array"]):
    # 2D data with default colormap
    ax[0].imshow(mapped, alpha=alpha)
    ax[0].set_title(f"2D data, alpha={alpha_type}")

    # 2D data with alpha-aware colormap
    ax[1].imshow(mapped, cmap=cmap_with_alpha, alpha=alpha)
    ax[1].set_title(f"2D with alpha cmap, alpha={alpha_type}")

    # RGB image
    ax[2].imshow(rgb, alpha=alpha)
    ax[2].set_title(f"RGB image, alpha={alpha_type}")

    # RGBA image (existing alpha channel)
    ax[3].imshow(rgba, alpha=alpha)
    ax[3].set_title(f"RGBA image, alpha={alpha_type}")

plt.suptitle("Alpha parameter behavior with different image types", fontsize=14)
plt.show()
