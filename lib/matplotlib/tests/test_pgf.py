import pytest

import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf


@pytest.mark.skipif(not matplotlib.checkdep_usetex(True),
                    reason='Missing TeX or Ghostscript or dvipng')
@image_comparison(baseline_images=['test_color_graphicx'],
                  extensions=['pdf', 'eps'],
                  tol=0.3)
def test_color_graphicx():
    # github issue #10042
    # if this test runs without crashing, that is considered success
    # image comparison succeeding is a bonus

    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['pgf.rcfonts'] = False

    plt.plot([1,2,3,4])
    plt.text(1, 1, r'\textcolor{red}{$y=x$}\rotatebox{10}{rotated}')
