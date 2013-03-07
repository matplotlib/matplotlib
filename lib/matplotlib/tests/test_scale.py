from __future__ import print_function

from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt
import numpy as np
import io


@image_comparison(baseline_images=['log_scales'], remove_text=True)
def test_log_scales():
    ax = plt.subplot(122, yscale='log', xscale='symlog')

    ax.axvline(24.1)
    ax.axhline(24.1)


@cleanup
def test_log_scatter():
    """Issue #1799"""
    fig, ax = plt.subplots(1)

    x = np.arange(10)
    y = np.arange(10) - 1

    ax.scatter(x, y)

    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')

    buf = io.BytesIO()
    fig.savefig(buf, format='eps')

    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
