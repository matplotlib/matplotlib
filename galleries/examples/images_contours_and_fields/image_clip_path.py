"""
============================
Clipping images with patches
============================

Demo of image that's been clipped by a circular patch.
"""

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.cbook as cbook
import matplotlib.patches as patches

image = Image.open(cbook.get_sample_data("grace_hopper.jpg"))

fig, ax = plt.subplots()
im = ax.imshow(image)
patch = patches.Circle((260, 200), radius=200, transform=ax.transData)
im.set_clip_path(patch)

ax.axis('off')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.artist.Artist.set_clip_path`
