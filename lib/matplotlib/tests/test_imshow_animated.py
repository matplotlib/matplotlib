import numpy as np
import matplotlib.pyplot as plt


def test_imshow_respects_animated_flag():
    fig, ax = plt.subplots()

    data = np.array([[1, 2], [3, 4]])

    # Animated image case
    im1 = ax.imshow(data, animated=True)
    assert im1.get_animated() is True

    # Animated images should not be registered in ax.images
    assert im1 not in ax.images

    # But the artist should still be attached to the Axes
    assert im1 in ax.get_children()

    # Non-animated image case
    im2 = ax.imshow(data, animated=False)
    assert im2.get_animated() is False

    # Non-animated images should be in ax.images by default
    assert im2 in ax.images

    # Use the figure so it is not flagged as unused in static analysis
    fig.canvas.draw()

