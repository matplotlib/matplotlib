
from matplotlib.testing.decorators import image_comparison
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt

@image_comparison(baseline_images=['imagegrid_cbar_mode'],
                  extensions=['png'],
                  remove_text=True)
def test_imagegrid_cbar_mode_edge():
    X, Y = np.meshgrid(np.linspace(0, 6, 30), np.linspace(0, 6, 30))
    arr = np.sin(X) * np.cos(Y) + 1j*(np.sin(3*Y) * np.cos(Y/2.))

    fig = plt.figure(figsize=(6, 6))

    grid = ImageGrid(fig, 111,
                          nrows_ncols = (2, 2),
                          direction='row',
                          cbar_location='right',
                          cbar_mode='edge',
                     )
    ax1, ax2, ax3, ax4, = grid

    im1 = ax1.imshow(arr.real, cmap='spectral')
    im2 = ax2.imshow(arr.imag, cmap='hot')
    im3 = ax3.imshow(np.abs(arr), cmap='jet')
    im4 = ax4.imshow(np.arctan2(arr.imag, arr.real), cmap='hsv')

    ax2.cax.colorbar(im2)
    ax4.cax.colorbar(im4)
