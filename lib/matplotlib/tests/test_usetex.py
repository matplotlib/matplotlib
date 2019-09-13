import pytest

import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


@pytest.fixture(autouse=True)  # All tests in this module use usetex.
def usetex():
    if not mpl.checkdep_usetex(True):
        pytest.skip('Missing TeX of Ghostscript or dvipng')
    mpl.rcParams['text.usetex'] = True


@image_comparison(baseline_images=['test_usetex'],
                  extensions=['pdf', 'png'],
                  tol=0.3)
def test_usetex():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.2,
            # the \LaTeX macro exercises character sizing and placement,
            # \left[ ... \right\} draw some variable-height characters,
            # \sqrt and \frac draw horizontal rules, \mathrm changes the font
            r'\LaTeX\ $\left[\int\limits_e^{2e}'
            r'\sqrt\frac{\log^3 x}{x}\,\mathrm{d}x \right\}$',
            fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])


@check_figures_equal()
def test_unicode_minus(fig_test, fig_ref):
    fig_test.text(.5, .5, "$-$")
    fig_ref.text(.5, .5, "\N{MINUS SIGN}")
