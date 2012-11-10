import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable, host_subplot
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from matplotlib.testing.decorators import image_comparison


def get_demo_image():
    """
    Load image used for tests.
    """
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)

def get_random_data():
    """
    Load "random" data used for tests.
    """
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/random_xy.npy", asfileobj=False)
    [x, y] = np.load(f)
    # z is a numpy array of 15x15
    return x, y

def add_sizebar(ax, size):
    asb =  AnchoredSizeBar(ax.transData,
                           size,
                           str(size),
                           loc=8,
                           pad=0.1, borderpad=0.5, sep=5,
                           frameon=False)
    ax.add_artist(asb)

def get_rgb():
    Z, extent = get_demo_image()
    Z[Z<0] = 0.
    Z = Z/Z.max()
    R = Z[:13,:13]
    G = Z[2:,2:]
    B = Z[:13,2:]
    return R, G, B


@image_comparison(baseline_images=['imagegrid'])
def test_imagegrid():
    """Test imagegrid"""

    F = plt.figure(1, (5.5, 3.5))
    grid = ImageGrid(F, 111, # similar to subplot(111)
                     nrows_ncols = (1, 3),
                     axes_pad = 0.1,
                     add_all=True,
                     label_mode = "L",
                 )

    Z, extent = get_demo_image() # demo image

    im1 = Z
    im2 = Z[:, :10]
    im3 = Z[:, 10:]
    vmin, vmax = Z.min(), Z.max()
    for i, im in enumerate([im1, im2, im3]):
        ax = grid[i]
        ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax,
                  interpolation="nearest")

@image_comparison(baseline_images=['simple_grid'])
def test_simple_grid():
    """
    A grid of 2x2 images with 0.05 inch pad between images and only
    the lower-left axes is labeled.
    """
    fig = plt.figure(1, (5.5, 3.5))
    grid = AxesGrid(fig, 111, # similar to subplot(131)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.05,
                    label_mode = "1",
                    )

    Z, extent = get_demo_image()
    for i in range(4):
        grid[i].imshow(Z, extent=extent, interpolation="nearest")

    # This only affects axes in first column and second row as
    # share_all = False.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

@image_comparison(baseline_images=['grid_with_single_cbar'])
def test_grid_with_single_cbar():
    """
    A grid of 2x2 images with a single colorbar
    """
    fig = plt.figure(1, (5.5, 3.5))
    grid = AxesGrid(fig, 111, # similar to subplot(132)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.0,
                    share_all = True,
                    label_mode = "L",
                    cbar_location = "top",
                    cbar_mode = "single",
                    )

    Z, extent = get_demo_image()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    # This affects all axes as share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

@image_comparison(baseline_images=['grid_with_each_cbar'])
def test_grid_with_each_cbar():
    """
    A grid of 2x2 images. Each image has its own colorbar.
    """
    fig = plt.figure(1, (5.5, 3.5))
    grid = AxesGrid(fig, 111, # similar to subplot(122)
                    nrows_ncols = (2, 2),
                    axes_pad = 0.1,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="top",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    Z, extent = get_demo_image()
    for i in range(4):
        im = grid[i].imshow(Z, extent=extent, interpolation="nearest")
        grid.cbar_axes[i].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])

@image_comparison(baseline_images=['scatter'])
def test_scatter():
    # the random data
    x, y = get_random_data()

    fig = plt.figure(1, figsize=(5.5, 5.5))

    # the scatter plot:
    axScatter = plt.subplot(111)
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 50, 100])

    #axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 50, 100])

@image_comparison(baseline_images=['ParasiteAxes_twinx'])
def test_ParasiteAxes_twinx():

    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Distance")
    host.set_ylabel("Density")
    par.set_ylabel("Temperature")

    p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")
    p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")

    leg = plt.legend()

    host.yaxis.get_label().set_color(p1.get_color())
    leg.texts[0].set_color(p1.get_color())

    par.yaxis.get_label().set_color(p2.get_color())
    leg.texts[1].set_color(p2.get_color())

@image_comparison(baseline_images=['ParasiteAxes_twin'])
def test_ParasiteAxes_twin():
    obs = [["01_S1", 3.88, 0.14, 1970, 63],
           ["01_S4", 5.6, 0.82, 1622, 150],
           ["02_S1", 2.4, 0.54, 1570, 40],
           ["03_S1", 4.1, 0.62, 2380, 170]]


    fig = plt.figure()

    ax_kms = SubplotHost(fig, 1, 1, 1, aspect=1.)

    # angular proper motion("/yr) to linear velocity(km/s) at distance=2.3kpc
    pm_to_kms = 1./206265.*2300*3.085e18/3.15e7/1.e5

    aux_trans = mtransforms.Affine2D().scale(pm_to_kms, 1.)
    ax_pm = ax_kms.twin(aux_trans)
    ax_pm.set_viewlim_mode("transform")

    fig.add_subplot(ax_kms)

    for n, ds, dse, w, we in obs:
        time = ((2007+(10. + 4/30.)/12)-1988.5)
        v = ds / time * pm_to_kms
        ve = dse / time * pm_to_kms
        ax_kms.errorbar([v], [w], xerr=[ve], yerr=[we], color="k", label=n)

    ax_kms.axis["bottom"].set_label("Linear velocity at 2.3 kpc [km/s]")
    ax_kms.axis["left"].set_label("FWHM [km/s]")
    ax_pm.axis["top"].set_label("Proper Motion [$^{''}$/yr]")
    ax_pm.axis["top"].label.set_visible(True)
    ax_pm.axis["right"].major_ticklabels.set_visible(False)

    ax_kms.set_xlim(950, 3700)
    ax_kms.set_ylim(950, 3100)
    # xlim and ylim of ax_pms will be automatically adjusted.

@image_comparison(baseline_images=['InsetLocator'])
def test_InsetLocator():
    fig = plt.figure(1, [5.5, 3])

    # first subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect(1.)

    axins = inset_axes(ax,
                       width="30%", # width = 30% of parent_bbox
                       height=1., # height : 1 inch
                       loc=3)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # second subplot
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect(1.)

    axins = zoomed_inset_axes(ax, 0.5, loc=1) # zoom = 0.5

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    add_sizebar(ax, 0.5)
    add_sizebar(axins, 0.5)

@image_comparison(baseline_images=['ZoomedInsetLocator'])
def test_ZoomedInsetLocator():
    fig = plt.figure(1, [5, 4])
    ax = fig.add_subplot(111)

    # prepare the demo image
    Z, extent = get_demo_image()
    Z2 = np.zeros([150, 150], dtype="d")
    ny, nx = Z.shape
    Z2[30:30+ny, 30:30+nx] = Z

    # extent = [-3, 4, -4, 3]
    ax.imshow(Z2, extent=extent, interpolation="nearest",
              origin="lower")

    axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
    axins.imshow(Z2, extent=extent, interpolation="nearest",
                 origin="lower")

    # sub region of the original image
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


@image_comparison(baseline_images=['RGBAxes'])
def test_RGBAxes():
    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

    r, g, b = get_rgb()
    kwargs = dict(origin="lower", interpolation="nearest")
    ax.imshow_rgb(r, g, b, **kwargs)

    ax.RGB.set_xlim(0., 9.5)
    ax.RGB.set_ylim(0.9, 10.6)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
