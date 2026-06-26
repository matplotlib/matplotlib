import os

import pytest

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison


def _test_agg_filter_alpha_impl():
    ax = plt.axes()
    x, y = np.mgrid[0:7, 0:8]
    data = x**2 - y**2
    mesh = ax.pcolormesh(data, cmap='Reds', zorder=5)

    def manual_alpha(im, dpi):
        im[:, :, 3] *= 0.6
        print('CALLED')
        return im, 0, 0

    # Note: Doing alpha like this is not the same as setting alpha on
    # the mesh itself. Currently meshes are drawn as independent patches,
    # and we see fine borders around the blocks of color. See the SO
    # question for an example: https://stackoverflow.com/q/20678817/
    mesh.set_agg_filter(manual_alpha)

    # Currently we must enable rasterization for this to have an effect in
    # the PDF backend.
    mesh.set_rasterized(True)

    ax.plot([0, 4, 7], [1, 3, 8])


@image_comparison(baseline_images=['agg_filter_alpha'],
                  extensions=['png', 'pdf'], style='mpl20')
def test_agg_filter_alpha():
    _test_agg_filter_alpha_impl()


# This test was broken out due to failing on Azure with py3.13
# This was the easiset way to XFail just a single image type
# Once that is sorted, this can return to being a single test with all extensions.
# And the impl method can be inlined again
@pytest.mark.xfail("TF_BUILD" in os.environ
                   and os.environ.get("VMIMAGE") == "windows-latest",
                   reason="Test failing on Azure under Python 3.13",
)
@image_comparison(baseline_images=['agg_filter_alpha'],
                  extensions=['gif'], style='mpl20')
def test_agg_filter_alpha_gif():
    _test_agg_filter_alpha_impl()
