"""
This test loads a subset of the files in Willem van Schaik's PNG test
suite available here:

   http://libpng.org/pub/png/pngsuite.html

The result should look like truth.png.
"""

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import glob

files = glob.glob("basn*.png")
files.sort()

plt.figure(figsize=(len(files), 2))

for i, fname in enumerate(files):
    data = plt.imread(fname)
    cmap = None # use default colormap
    if data.ndim==2:
        # keep grayscale images gray
        cmap = cm.gray
    plt.imshow(data, extent=[i,i+1,0,1], cmap=cmap)

plt.gca().get_frame().set_facecolor("#ddffff")
plt.gca().set_xlim(0, len(files))
plt.show()
