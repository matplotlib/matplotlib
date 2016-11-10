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


@image_comparison(baseline_images=['logit_scales'], remove_text=True,
                  extensions=['png'])
def test_logit_scales():
    ax = plt.subplot(111, xscale='logit')

    # Typical extinction curve for logit
    x = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.997, 0.999])
    y = 1.0 / x

    ax.plot(x, y)
    ax.grid(True)


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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
