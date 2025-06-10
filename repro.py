# Standalone script demonstrating three things:
# - Generates random 2D array (arr) and turns it into RGB via skimage.color.gray2rgb
# - Makes full-size 2D alpha array with, say, the top half at 0.2 transparency and the bottom half at 1.0
# - On left, calls imshow(arr, alpha=alpha, cmap='gray') 
# - On right, calls imshow(arr_rgb, alpha=alpha) 

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb

arr = np.random.random((10, 10))
arr_rgb = gray2rgb(arr)
alpha = np.ones_like(arr)
alpha[:5] 0.2

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Expected"); axl.imshow(arr, alpha=alpha, cmp='gray')
ax2.set_title("Broken"); ax2.imshow(arr_rgb, alpha=alpha)
plt.show()