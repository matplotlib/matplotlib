from matplotlib.testing.decorators import image_comparison, knownfailureif
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import glob
import os

@image_comparison(baseline_images=['pngsuite'], extensions=['png'])
def test_pngsuite():
    dirname = os.path.join(
        os.path.dirname(__file__),
        'baseline_images',
        'pngsuite')
    files = glob.glob(os.path.join(dirname, 'basn*.png'))
    files.sort()

    fig = plt.figure(figsize=(len(files), 2))

    for i, fname in enumerate(files):
        data = plt.imread(fname)
        cmap = None # use default colormap
        if data.ndim==2:
            # keep grayscale images gray
            cmap = cm.gray
        plt.imshow(data, extent=[i,i+1,0,1], cmap=cmap)

    plt.gca().get_frame().set_facecolor("#ddffff")
    plt.gca().set_xlim(0, len(files))
