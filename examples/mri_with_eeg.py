#!/usr/bin/env python
"""
This now uses the imshow command instead of pcolor which *is much
faster*
"""
from __future__ import division
from matplotlib.matlab import *
from matplotlib.lines import Line2D
from matplotlib.transforms import get_bbox_transform, Point, Value, Bbox,\
     translation_transform, zero
# I use if 1 to break up the different regions of code visually

if 1:   # load the data
    # data are 256x256 16 bit integers
    dfile = 'data/s1045.ima'
    im = fromstring(file(dfile, 'rb').read(), Int16).astype(Float)
    im.shape = 256, 256

if 1: # plot the MRI in pcolor
    subplot(221)
    imshow(im, ColormapJet(256))
    axis('off')

if 1:  # plot the histogram of MRI intensity
    subplot(222)
    im.shape = 256*256,
    im = take(im, nonzero(im)) # ignore the background
    im = im/(2.0**15) # normalize
    hist(im, 100)
    set(gca(), 'xticks', [-1, -.5, 0, .5, 1])
    set(gca(), 'yticks', [])
    xlabel('intensity')
    ylabel('MRI density')

if 1:   # plot the EEG
    # load the data
    numSamples, numRows = 800,4
    data = fromstring(file('data/eeg.dat', 'rb').read(), Float)
    data.shape = numSamples, numRows
    t = arange(numSamples)/float(numSamples)*10.0
    ticklocs = []
    ax = subplot(212)

    height = 72  # height of one EEG in pixels
    # transform data to axes coord (0,1)
    

    for i in range(numRows):
        trans = get_bbox_transform(ax.viewLim, ax.bbox)
        offset = (i+1)/(numRows+1)
        height = Value(offset) * (ax.bbox.ur().y() - ax.bbox.ll().y())
        trans.set_offset( (0,0), translation_transform(zero(), height) )
        
        thisLine = Line2D(
            t, data[:,i]-data[0,i],
            )
        
        thisLine.set_transform(trans)
        
        ax.add_line(thisLine)
        ticklocs.append(offset)

    set(gca(), 'xlim', [0,10])
    set(gca(), 'ylim', [0,200])    
    set(gca(), 'xticks', arange(10))
    yticks = set(gca(), 'yticks', ticklocs)
    set(gca(), 'yticklabels', ['PG3', 'PG5', 'PG7', 'PG9']) 

    # set the yticks to use axes coords on the y axis
    set(yticks, 'transform', ax.transAxes)
    xlabel('time (s)')


#savefig('mri_with_eeg')
show()
