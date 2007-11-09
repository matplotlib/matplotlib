#!/usr/bin/env python
"""
This now uses the imshow command instead of pcolor which *is much
faster*
"""
from __future__ import division
from pylab import *
from matplotlib.lines import Line2D
from matplotlib.transforms import get_bbox_transform, Point, Value, Bbox,\
     unit_bbox


# I use if 1 to break up the different regions of code visually

if 1:   # load the data
    # data are 256x256 16 bit integers
    dfile = 'data/s1045.ima'
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
    xticks([-1, -.5, 0, .5, 1])
    yticks([])
    xlabel('intensity')
    ylabel('MRI density')

if 1:   # plot the EEG
    # load the data
    numSamples, numRows = 800,4
    data = fromstring(file('data/eeg.dat', 'rb').read(), float)
    data.shape = numSamples, numRows
    t = arange(numSamples)/float(numSamples)*10.0
    ticklocs = []
    ax = subplot(212)

    boxin = Bbox(
        Point(ax.viewLim.ll().x(), Value(-20)),
        Point(ax.viewLim.ur().x(), Value(20)))


    height = ax.bbox.ur().y() - ax.bbox.ll().y()
    boxout = Bbox(
        Point(ax.bbox.ll().x(), Value(-1)*height),
        Point(ax.bbox.ur().x(), Value(1) * height))


    transOffset = get_bbox_transform(
        unit_bbox(),
        Bbox( Point( Value(0), ax.bbox.ll().y()),
              Point( Value(1), ax.bbox.ur().y())
              ))


    for i in range(numRows):
        # effectively a copy of transData
        trans = get_bbox_transform(boxin, boxout)
        offset = (i+1)/(numRows+1)

        trans.set_offset( (0, offset), transOffset)

        thisLine = Line2D(
            t, data[:,i]-data[0,i],
            )

        thisLine.set_transform(trans)

        ax.add_line(thisLine)
        ticklocs.append(offset)

    xlim(0,10)
    xticks(arange(10))

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


#savefig('mri_with_eeg')
show()
