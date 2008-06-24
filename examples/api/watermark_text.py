"""
Use a PNG file as a watermark
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('image', origin='upper')

dpi = 100   # save dpi
w, h = 8, 6 # inches

parser = mathtext.MathTextParser("Bitmap")

rgba, depth1 = parser.to_rgba(r'Property of MPL', color='gray',
                              fontsize=30, dpi=200)
rgba[:,:,-1] *= 0.5
fig = plt.figure(figsize=(w,h))

ax = fig.add_subplot(111)
ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()

imh, imw, tmp = rgba.shape

# position image at bottom right
fig.figimage(rgba.astype(float)/255., w*dpi-imw, 0)


fig.savefig('watermarked_text', transparent=True, dpi=dpi)

