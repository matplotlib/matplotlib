import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['axes_titles'])
def test_axes_titles():
    # Related to issue #3327
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_title('center', loc='center', fontsize=20, fontweight=700)
    ax1.set_title('left', loc='left', fontsize=10, fontweight=400)
    ax2 = fig.add_subplot(212)
    ax2.set_title('left', loc='left', fontsize=10, fontweight=400)
    ax2.set_title('center', loc='center', fontsize=20, fontweight=700)
