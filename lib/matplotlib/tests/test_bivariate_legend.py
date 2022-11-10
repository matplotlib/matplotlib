import numpy as np
import matplotlib.axes
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

from matplotlib.testing.decorators import check_figures_equal

import matplotlib.bivariate_legend as mbvLegend
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm



# test to make sure modes work
def test_mode_value():
    """
    tests to make sure the modes work correctly
    """
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)

    fig = plt.figure()
    cax = fig.add_subplot()
    blegend = mbvLegend.Bivariate_legend(cax, x, y, mode='value')
    
    val_first = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend()[0][1].tolist())
    val_second = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend()[0][3].tolist())
    assert val_first[2] < val_second[2] and val_first[1] == val_second[1] and val_first[0] == val_second[0]

    
    # test both
    # test matplotlib colors

def test_mode_saturation():
    # test saturation
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(cax, x, y, mode='saturation')

    val_first = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend()[0][1].tolist())
    val_second = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend()[0][3].tolist())
    assert val_first[1] < val_second[1] 
    assert val_first[2] == val_second[2] 
    assert val_first[0] == val_second[0]

# test normalize
def test_norms():
    # test to make sure normalize is working
    # by using PowerNorm as a test
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    blegend_norm = mbvLegend.Bivariate_legend(
        cax, x, y, d1norm=norm)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y)
    colors_norm = mcolors.rgb_to_hsv(blegend_norm.get_mapped_colors())
    colors = mcolors.rgb_to_hsv(blegend.get_mapped_colors())
    assert colors_norm[1][0] < colors[1][0]

# test all 3 cmap options
def test_cmap_precanned():
    # test to make sure the user can use the 
    # 1 of the 9 precanned colormap options
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y, cmap = 'purple_cyan', mode='value')
    legend_colors = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend())
    assert legend_colors[1][0][0] == 0.5

def test_cmap_custom():
    # test for a cmap with any hsv color given
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y, cmap=[[0.7, 0.3, 0.2], [0.4, 0.8, 0.6]], mode='value')
    legend_colors = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend())
    assert legend_colors[1][0][0].round(1) == 0.4

def test_cmap_mpl():
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y, cmap=['viridis', 'plasma'], mode='value')
    legend_colors = mcolors.rgb_to_hsv(
        blegend._get_colors_in_bvlegend())

# test extends
    # set minimum & maximum on d1 and d2, and then test them
    # for extending min, max, and both
def test_extend_nobins():
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend_extend = mbvLegend.Bivariate_legend(
        cax, x, y, d1minimum=0, d1maximum=2, d1extend='both')
    f1, cax1 = plt.subplots()
    blegend = mbvLegend.Bivariate_legend(
        cax1, x, y)
    colors_extend = mcolors.rgb_to_hsv(blegend_extend.get_mapped_colors())
    colors = mcolors.rgb_to_hsv(blegend.get_mapped_colors())
    assert colors[1][0] > colors_extend[1][0]
    # check to make sure the ticks are correct
    ticks = blegend_extend.get_bvlegend_axes().get_xticklabels()
    tick_labels = [ticks[i].get_text() for i in range(len(ticks))]
    assert tick_labels == ['0.0', '0.5', '1.0', '1.5', '2.0']

# test extends with bins (also a test for num_bins)
def test_extend_bins():
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend_extend = mbvLegend.Bivariate_legend(
        cax, x, y, d1minimum=0, d1maximum=2, d1extend='both', d1num_bins=4, d2num_bins=4)
    colors = mcolors.rgb_to_hsv(blegend_extend._get_colors_in_bvlegend())
    bins = len(set(colors[0][:,1].tolist())) - 1
    assert bins == 6

# test how a user can send in their own bins
def test_custom_bins():
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y, d1num_bins=4, d2num_bins=4, d1bins=[-3, -2, 1.2, 2.5, 4.0])
    ticks = blegend.get_bvlegend_axes().get_xticklabels()
    tick_labels = [float(ticks[i].get_text()) for i in range(len(ticks))]
    assert tick_labels == [-3, -2, 1.2, 2.5, 4.0]
    colors = mcolors.rgb_to_hsv(blegend._get_colors_in_bvlegend())
    bins = len(set(colors[0][:, 1].tolist())) - 1
    assert bins == 4

# test ticks (when a user send in their own ticks)
def test_custom_ticks():
    f, cax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend = mbvLegend.Bivariate_legend(
        cax, x, y, d1ticks=[0, 0.25, 0.6, 0.9, 1])
    ticks = blegend.get_bvlegend_axes().get_xticks().tolist()
    assert ticks == [0.0, 0.25, 0.6, 0.9, 1.0]

# test tick rotation


@check_figures_equal(extensions=["png"])
def test_legend_displayed(fig_ref, fig_test):
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    cax1 = fig_ref.add_subplot()
    blegend = mbvLegend.Bivariate_legend(
        cax1, x, y)
    cax2 = fig_test.add_subplot()
    cax2.imshow(blegend._get_colors_in_bvlegend(),
                origin="lower", extent=[0, 1, 0, 1])
    cax2.set_title('Legend', fontsize=10)
    cax2.set_xlabel("Variable 1")
    cax2.set_ylabel("Variable 2")
    cax2.set_xticks(blegend.get_bvlegend_axes().get_xticks())
    cax2.set_yticks(blegend.get_bvlegend_axes().get_yticks())
    cax2.set_xticklabels(blegend.get_bvlegend_axes().get_xticklabels())
    cax2.set_yticklabels(blegend.get_bvlegend_axes().get_yticklabels())

# test to make sure tick rotation is working
@check_figures_equal(extensions=["png"])
def test_tick_rotation(fig_ref, fig_test):
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    cax1 = fig_ref.add_subplot()
    blegend = mbvLegend.Bivariate_legend(
        cax1, x, y, d1tick_rotation=30)
    cax2 = fig_test.add_subplot()
    cax2.imshow(blegend._get_colors_in_bvlegend(),
                origin="lower", extent=[0, 1, 0, 1])
    cax2.set_title('Legend', fontsize=10)
    cax2.set_xlabel("Variable 1")
    cax2.set_ylabel("Variable 2")
    cax2.set_xticks(blegend.get_bvlegend_axes().get_xticks())
    cax2.set_yticks(blegend.get_bvlegend_axes().get_yticks())
    cax2.set_xticklabels(blegend.get_bvlegend_axes().get_xticklabels(), rotation=30)
    cax2.set_yticklabels(blegend.get_bvlegend_axes().get_yticklabels())

# test to make sure white is darkened if specified 
# (works only when mode is 'saturation' or 'both')
def test_darken_white():
    f, cax = plt.subplots()
    f1, cax1 = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    blegend_dark = mbvLegend.Bivariate_legend(
        cax, x, y, d1num_bins=4, d2num_bins=4, darkenWhite=True)
    blegend = mbvLegend.Bivariate_legend(
        cax1, x, y, d1num_bins=4, d2num_bins=4)
    dark = mcolors.rgb_to_hsv(blegend_dark._get_colors_in_bvlegend())
    normal = mcolors.rgb_to_hsv(blegend._get_colors_in_bvlegend())
    assert dark[0][:, 2][0] < normal[0][:, 2][0]

# check to make sure imshow kwargs can be passed along
@check_figures_equal(extensions=["png"])
def test_imshow_kwargs(fig_ref, fig_test):
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    cax1 = fig_ref.add_subplot()
    blegend = mbvLegend.Bivariate_legend(
        cax1, x, y, interpolation='nearest', alpha=1)
    cax2 = fig_test.add_subplot()
    cax2.imshow(blegend._get_colors_in_bvlegend(),
                origin="lower", extent=[0, 1, 0, 1], interpolation='nearest', alpha=1)
    cax2.set_title('Legend', fontsize=10)
    cax2.set_xlabel("Variable 1")
    cax2.set_ylabel("Variable 2")
    cax2.set_xticks(blegend.get_bvlegend_axes().get_xticks())
    cax2.set_yticks(blegend.get_bvlegend_axes().get_yticks())
    cax2.set_xticklabels(blegend.get_bvlegend_axes().get_xticklabels())
    cax2.set_yticklabels(blegend.get_bvlegend_axes().get_yticklabels())
    
