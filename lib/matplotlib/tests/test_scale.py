from __future__ import print_function

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import numpy as np
import io


@image_comparison(baseline_images=['log_scales'], remove_text=True)
def test_log_scales():
    ax = plt.figure().add_subplot(122, yscale='log', xscale='symlog')

    ax.axvline(24.1)
    ax.axhline(24.1)


@image_comparison(baseline_images=['logit_scales'], remove_text=True,
                  extensions=['png'])
def test_logit_scales():
    ax = plt.figure().add_subplot(111, xscale='logit')

    # Typical extinction curve for logit
    x = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.997, 0.999])
    y = 1.0 / x

    ax.plot(x, y)
    ax.grid(True)


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


def test_logscale_subs():
    fig, ax = plt.subplots()
    ax.set_yscale('log', subsy=np.array([2, 3, 4]))
    # force draw
    fig.canvas.draw()


@image_comparison(baseline_images=['logscale_mask'], remove_text=True,
                  extensions=['png'])
def test_logscale_mask():
    # Check that zero values are masked correctly on log scales.
    # See github issue 8045
    xs = np.linspace(0, 50, 1001)

    fig, ax = plt.subplots()
    ax.plot(np.exp(-xs**2))
    fig.canvas.draw()
    ax.set(yscale="log")
