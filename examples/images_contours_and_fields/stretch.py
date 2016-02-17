
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.imageStretch as mstretch

def main():
    img = np.load("starfield.npy")

    ##Clean up image. Remove Nans, and negative values.
    #Negative values are unphysical, and can't be represented in
    #a log stretch or a square root stretch.
    nan = ~np.isfinite(img)
    medianValue =  np.median(img[~nan])
    img[nan] = medianValue
   
    img[img < 0] = medianValue

    subImg = img[1000:1601, 400:1001]
    
    
    show = lambda x: plt.imshow(x, interpolation="nearest", cmap=plt.cm.Greys, norm=norm, aspect="auto")
    
    plt.clf()
    plt.subplot(221)
    plt.cla()
    norm = mstretch.LinearStretch()
    show(subImg)
    plt.colorbar()
    plt.title("Linear Stretch")
    
    plt.subplot(222)
    plt.cla()
    norm = mstretch.LogStretch()
    show(subImg)
    plt.colorbar()
    plt.title("Log Stretch")
    
    plt.subplot(223)
    plt.cla()
    norm = mstretch.SqrtStretch()
    show(subImg)
    plt.colorbar()
    plt.title("Square Root Stretch")

    plt.subplot(224)
    plt.cla()
    norm = mstretch.HistEquStretch(img.flatten(), 5, 99)
    show(subImg)
    plt.colorbar()
    plt.title("Histogram Equalisation Stretch")
