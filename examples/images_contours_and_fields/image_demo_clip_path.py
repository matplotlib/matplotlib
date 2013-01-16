"""
Demo of image that's been clipped by a circular patch.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook


image_file = cbook.get_sample_data('lena.npy')
image = np.load(image_file)

fig, ax = plt.subplots()
im = ax.imshow(image)
patch = patches.Circle((130, 130), radius=100, transform=ax.transData)
im.set_clip_path(patch)

plt.axis('off')
plt.show()
