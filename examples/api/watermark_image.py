"""
Use a PNG file as a watermark
"""
import numpy as np
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt

datafile = cbook.get_sample_data('logo2.png', asfileobj=False)
print 'loading', datafile
im = image.imread(datafile)
im[:,:,-1] = 0.5  # set the alpha channel

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()
fig.figimage(im, 10, 10)

plt.show()
