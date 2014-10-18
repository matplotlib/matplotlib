"""
Illustrate some helper functions for shading regions where a logical
mask is True

See :meth:`matplotlib.collections.BrokenBarHCollection.span_where`
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections


def span_regions(ax, fig_title, x, y, plot_line_color='k', plot_line_weight=0.5, 
                 zero_line_color='k', zero_line_weight=0.5,
                 pos_rgn_color='green', pos_rgn_alpha=0.5,
                 neg_rgn_color='red', neg_rgn_alpha=0.5,
                 threshold=0):
    
    ax.set_title(fig_title)
    ax.plot(x, y, color=plot_line_color, lw=plot_line_weight)
    ax.axhline(0, color=zero_line_color, lw=zero_line_weight)
    
    collection = collections.BrokenBarHCollection.span_where(
        t, ymin=0, ymax=1, where=s > threshold, facecolor=pos_rgn_color, alpha=pos_rgn_alpha)
    ax.add_collection(collection)
    
    collection = collections.BrokenBarHCollection.span_where(
        t, ymin=-1, ymax=0, where=s < threshold, facecolor=neg_rgn_color, alpha=neg_rgn_alpha)
    ax.add_collection(collection)
        
    plt.show()

#--------------------------------------------------------------------------------------------------------
t = np.arange(0.0, 2, 0.01)
s = np.sin(2*np.pi*t)
fig, ax = plt.subplots()

span_regions(ax, "using span_where", t, s, plot_line_color='k', plot_line_weight=0.5, zero_line_color='k', 
             zero_line_weight=0.75, pos_rgn_color='green', neg_rgn_color='red', pos_rgn_alpha=0.75,
             neg_rgn_alpha=0.2, threshold=0)