import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

@check_figures_equal()
def test_polar_log_rorigin_rendering(fig_test, fig_ref):
    r = np.logspace(-1, 1, 500)
    theta = np.linspace(0, 2 * np.pi, 500)

    # Reference (correct rendering after fix)
    ax_ref = fig_ref.add_subplot(1, 1, 1, projection='polar')
    ax_ref.set_rscale('log')
    ax_ref.set_rorigin(-0.5)
    ax_ref.plot(theta, r)

    # Test output (same code, expected to match)
    ax_test = fig_test.add_subplot(1, 1, 1, projection='polar')
    ax_test.set_rscale('log')
    ax_test.set_rorigin(-0.5)
    ax_test.plot(theta, r)
