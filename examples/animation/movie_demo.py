#!/usr/bin/python
#
# Josh Lifton 2004
#
# Permission is hereby granted to use and abuse this document
# so long as proper attribution is given.
#
# This Python script demonstrates how to use the numarray package
# to generate and handle large arrays of data and how to use the
# matplotlib package to generate plots from the data and then save
# those plots as images.  These images are then stitched together
# by Mencoder to create a movie of the plotted data.  This script
# is for demonstration purposes only and is not intended to be
# for general use.  In particular, you will likely need to modify
# the script to suit your own needs.
#


from matplotlib.matlab import *   # For plotting graphs.
import os                         # For issuing commands to the OS.
import sys                        # For determining the Python version.

#
# Print the version information for the machine, OS,
# Python interpreter, and matplotlib.  The version of
# Mencoder is printed when it is called.
#
# This script is known to have worked for:
#
# OS version: ('Linux', 'flux-capacitor', '2.4.26', '#1 SMP Sa Apr 17 19:33:42 CEST 2004', 'i686')
# Python version: 2.3.4 (#2, May 29 2004, 03:31:27) [GCC 3.3.3 (Debian 20040417)]
# matplotlib version: 0.61.0
# MEncoder version:
# MEncoder 1.0pre4-3.3.3 (C) 2000-2004 MPlayer Team
# CPU: Intel Celeron 2/Pentium III Coppermine,Geyserville 996.1 MHz (Family: 6, Stepping: 10)
# Detected cache-line size is 32 bytes
# CPUflags: Type: 6 MMX: 1 MMX2: 1 3DNow: 0 3DNow2: 0 SSE: 1 SSE2: 0
# Compiled for x86 CPU with extensions: MMX MMX2 SSE
#
print 'Executing on', os.uname()
print 'Python version', sys.version
print 'matplotlib version', matplotlib.__version__



#
# First, let's create some data to work with.  In this example
# we'll use a normalized Gaussian waveform whose mean and
# standard deviation both increase linearly with time.  Such a
# waveform can be thought of as a propagating system that loses
# coherence over time, as might happen to the probability
# distribution of a clock subjected to independent, identically
# distributed Gaussian noise at each time step.
#

print 'Initializing data set...'   # Let the user know what's happening.

# Initialize variables needed to create and store the example data set.
numberOfTimeSteps = 100   # Number of frames we want in the movie.
x = arange(-10,10,0.01)   # Values to be plotted on the x-axis.
mean = -6                 # Initial mean of the Gaussian.
stddev = 0.2              # Initial standard deviation.
meaninc = 0.1             # Mean increment.
stddevinc = 0.1           # Standard deviation increment.

# Create an array of zeros and fill it with the example data.
y = zeros((numberOfTimeSteps,len(x)), Float64)  
for i in range(numberOfTimeSteps) :
    y[i] = (1/sqrt(2*pi*stddev))*exp(-((x-mean)**2)/(2*stddev))
    mean = mean + meaninc
    stddev = stddev + stddevinc

print 'Done.'                       # Let the user know what's happening.

#
# Now that we have an example data set (x,y) to work with, we can
# start graphing it and saving the images.
#

for i in range(len(y)) :
    #
    # The next four lines are just like Matlab.
    #
    plot(x,y[i],'b.')
    axis((x[0],x[-1],-0.25,1))
    xlabel('time (ms)')
    ylabel('probability density function')
    
    #
    # Notice the use of LaTeX-like markup.
    #
    title(r'$\cal{N}(\mu, \sigma^2)$', fontsize=20)

    #
    # The file name indicates how the image will be saved and the
    # order it will appear in the movie.  If you actually wanted each
    # graph to be displayed on the screen, you would include commands
    # such as show() and draw() here.  See the matplotlib
    # documentation for details.  In this case, we are saving the
    # images directly to a file without displaying them.
    #
    filename = str('%03d' % i) + '.png'
    savefig(filename, dpi=100)

    #
    # Let the user know what's happening.
    #
    print 'Wrote file', filename

    #
    # Clear the figure to make way for the next image.
    #
    clf()

#
# Now that we have graphed images of the dataset, we will stitch them
# together using Mencoder to create a movie.  Each image will become
# a single frame in the movie.
#
# We want to use Python to make what would normally be a command line
# call to Mencoder.  Specifically, the command line call we want to
# emulate is (without the initial '#'):
# mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi
# See the MPlayer and Mencoder documentation for details.
#

command = ('mencoder',
           'mf://*.png',
           '-mf',
           'type=png:w=800:h=600:fps=25',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           'output.avi')

os.spawnvp(os.P_WAIT, 'mencoder', command)
