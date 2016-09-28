"""
This now uses the imshow command instead of pcolor which *is much
faster*
"""

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm

from matplotlib.collections import LineCollection
# I use if 1 to break up the different regions of code visually

fig = plt.figure("MRI_with_EEG")

if 1:   # load the data
    # data are 256x256 16 bit integers
    dfile = cbook.get_sample_data('s1045.ima.gz')
    im = np.fromstring(dfile.read(), np.uint16).astype(float)
    im.shape = (256, 256)

if 1:  # plot the MRI in pcolor
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.imshow(im, cmap=cm.gray)
    ax0.axis('off')

if 1:  # plot the histogram of MRI intensity
    ax1 = fig.add_subplot(2, 2, 2)
    im = np.ravel(im)
    im = im[np.nonzero(im)]  # ignore the background
    im = im / (2**15)  # normalize
    ax1.hist(im, 100)
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticks([])
    ax1.set_xlabel('intensity')
    ax1.set_ylabel('MRI density')

if 1:   # plot the EEG
    # load the data

    numSamples, numRows = 800, 4
    eegfile = cbook.get_sample_data('eeg.dat', asfileobj=False)
    print('loading eeg %s' % eegfile)
    data = np.fromstring(open(eegfile, 'rb').read(), float)
    data.shape = (numSamples, numRows)
    t = 10.0 * np.arange(numSamples) / numSamples
    ticklocs = []
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlim(0, 10)
    ax2.set_xticks(np.arange(10))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin) * 0.7  # Crowd them a bit.
    y0 = dmin
    y1 = (numRows - 1) * dr + dmax
    ax2.set_ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:, np.newaxis], data[:, i, np.newaxis])))
        ticklocs.append(i * dr)

    offsets = np.zeros((numRows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    lines = LineCollection(segs, offsets=offsets, transOffset=None)
    ax2.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax2.set_yticks(ticklocs)
    ax2.set_yticklabels(['PG3', 'PG5', 'PG7', 'PG9'])

    ax2.set_xlabel('time (s)')

plt.show()
