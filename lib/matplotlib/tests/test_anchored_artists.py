import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import anchored_artists
from matplotlib.testing.decorators import image_comparison
import matplotlib.font_manager as fm

@image_comparison(baseline_images=['anchored_sizebar'])
def test_anchored_sizebar():
    fig, ax = plt.subplots()
    ax.imshow(np.arange(100).reshape(10,10))
    fontprops = fm.FontProperties(size=20, family='monospace')
    bar = anchored_artists.AnchoredSizeBar(ax.transData, 3, '3 units', 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.5, color='white', fontprops=fontprops)
    ax.add_artist(bar)
