import math

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['polar_proj'],extensions=['png'])
def test_adding_rectangle_patch():
    fig = plt.figure(figsize=(640/80,640/80), dpi = 80)
    ax = fig.add_subplot(111, projection='polar')

    # add quadrant as example
    ax.add_patch(
        patches.Rectangle(
            (0, 1), width=math.pi * 0.5, height=0.5
        )
    )
    ax.bar(0, 1).remove()
    ax.set_rmax(2)
