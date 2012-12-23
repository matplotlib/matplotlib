"""
Simple demo of the imshow function.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

image_file = cbook.get_sample_data('lena.npy')
image = np.load(image_file)

plt.imshow(image)
plt.axis('off') # clear x- and y-axes
plt.show()

