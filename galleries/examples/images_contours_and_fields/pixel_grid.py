"""
=========================
Image with Pixel Grid Overlay
=========================

This example shows how to overlay a pixel-aligned grid on an image in Matplotlib.
Such grids can be useful for teaching, image analysis, or visual inspection.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def pixel_grid(image, cell_size=50, grid_color="gray", grid_width=0.5):
    """
    Display an image with a pixel grid overlay.

    Parameters
    ----------
    image : str or ndarray
        Path to an image file or a NumPy array representing the image.
    cell_size : int, default=50
        Spacing between grid lines, in pixels.
    grid_color : str, default='gray'
        Color of the grid lines.
    grid_width : float, default=0.5
        Thickness of the grid lines.
    """
    # Load image from file or array
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = Image.fromarray(image)

    width, height = img.size

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    # Vertical grid lines
    for x in range(0, width, cell_size):
        ax.axvline(x, color=grid_color, linewidth=grid_width)

    # Horizontal grid lines
    for y in range(0, height, cell_size):
        ax.axhline(y, color=grid_color, linewidth=grid_width)

    # Set limits and aspect ratio
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])  # flip Y-axis
    ax.set_aspect("equal")

    ax.set_title("Image with Pixel Grid Overlay")

    plt.show()


# Example usage with synthetic image
if __name__ == "__main__":
    # Create a synthetic grayscale image (200x300 pixels)
    img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

    pixel_grid(img, cell_size=40, grid_color="red", grid_width=1.0)