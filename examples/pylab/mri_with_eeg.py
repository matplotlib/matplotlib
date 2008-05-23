#!/usr/bin/env python
"""
This now uses the imshow command instead of pcolor which *is much
faster*
"""
from __future__ import division
from pylab import *
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, BboxTransform, BboxTransformTo, Affine2D

# I use if 1 to break up the different regions of code visually

if 1:   # load the data
    # data are 256x256 16 bit integers
    dfile = '../data/s1045.ima'
    im = fromstring(file(dfile, 'rb').read(), uint16).astype(float)
    im.shape = 256, 256

if 1: # plot the MRI in pcolor
    subplot(221)
    imshow(im, cmap=cm.jet)
    axis('off')

if 1:  # plot the histogram of MRI intensity
    subplot(222)
    im = ravel(im)
    im = take(im, nonzero(im)) # ignore the background
    im = im/(2.0**15) # normalize
    hist(im, 100)
    print im.shape
    xticks([-1, -.5, 0, .5, 1])
    yticks([])
    xlabel('intensity')
    ylabel('MRI density')

if 1:   # plot the EEG
    # load the data

    numSamples, numRows = 800,4
    data = fromstring(file('../data/eeg.dat', 'rb').read(), float)
    data.shape = numSamples, numRows
    t = arange(numSamples)/float(numSamples)*10.0
    ticklocs = []
    ax = subplot(212)
    xlim(0,10)
    xticks(arange(10))

    boxin = Bbox.from_extents(ax.viewLim.x0, -20, ax.viewLim.x1, 20)

    height = ax.bbox.height
    boxout = Bbox.from_extents(ax.bbox.x0, -1.0 * height,
                               ax.bbox.x1,  1.0 * height)

    transOffset = BboxTransformTo(
        Bbox.from_extents(0.0, ax.bbox.y0, 1.0, ax.bbox.y1))


    for i in range(numRows):
        # effectively a copy of transData
        trans = BboxTransform(boxin, boxout)
        offset = (i+1)/(numRows+1)

        trans += Affine2D().translate(*transOffset.transform_point((0, offset)))

        thisLine = Line2D(
            t, data[:,i]-data[0,i],
            )

        thisLine.set_transform(trans)

        ax.add_line(thisLine)
        ticklocs.append(offset)

    setp(gca(), 'yticklabels', ['PG3', 'PG5', 'PG7', 'PG9'])

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_transform(ax.transAxes)
        tick.label2.set_transform(ax.transAxes)
        tick.tick1line.set_transform(ax.transAxes)
        tick.tick2line.set_transform(ax.transAxes)
        tick.gridline.set_transform(ax.transAxes)


    xlabel('time (s)')

if 1:
    savefig('mri_with_eeg')

show()
