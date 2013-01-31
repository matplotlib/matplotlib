"""
Tests specific to the lines module.
"""

from nose.tools import assert_true
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup, image_comparison

@cleanup
def test_set_line_coll_dash():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    np.random.seed(0)
    # Testing setting linestyles for line collections.
    # This should not produce an error.
    cs = ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])

    assert True


@image_comparison(baseline_images=['line_collection_dashes'], remove_text=True)
def test_set_line_coll_dash_image():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    np.random.seed(0)
    cs = ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])
