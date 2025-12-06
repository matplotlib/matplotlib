from matplotlib.testing.decorators import check_figures_equal
import numpy as np


@check_figures_equal()
def test_imshow_respects_animated(fig_test, fig_ref):
    rng = np.random.default_rng(19680801)
    data = rng.random((8, 8))

    ax_t = fig_test.subplots()
    ax_t.imshow(data, animated=True)  # should be skipped on initial draw

    fig_ref.subplots()  # blank reference
