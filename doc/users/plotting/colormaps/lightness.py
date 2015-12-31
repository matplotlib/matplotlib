'''
For each colormap, plot the lightness parameter L* from CIELAB colorspace
along the y axis vs index through the colormap. Colormaps are examined in
categories as in the original matplotlib gallery of colormaps.
'''

from colormaps import cmaps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from colorspacious import cspace_converter

mpl.rcParams.update({'font.size': 12})

# indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
for cmap_category, cmap_list in cmaps:

    # Do subplots so that colormaps have enough space. 5 per subplot?
    dsub = 5 # number of colormaps per subplot
    if cmap_category == 'Diverging': # because has 12 colormaps
        dsub = 6
    elif cmap_category == 'Sequential (2)':
        dsub = 6
    elif cmap_category == 'Sequential':
        dsub = 7
    nsubplots = int(np.ceil(len(cmap_list)/float(dsub)))

    fig = plt.figure(figsize=(7,2.6*nsubplots))

    for i, subplot in enumerate(range(nsubplots)):

        locs = [] # locations for text labels

        ax = fig.add_subplot(nsubplots, 1, i+1)

        for j, cmap in enumerate(cmap_list[i*dsub:(i+1)*dsub]):

            # Get rgb values for colormap
            rgb = cm.get_cmap(cmap)(x)[np.newaxis,:,:3]

            # Get colormap in CAM02-UCS colorspace. We want the lightness.
            lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

            # Plot colormap L values
            # Do separately for each category so each plot can be pretty
            # to make scatter markers change color along plot:
            # http://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
            if cmap_category=='Perceptually Uniform Sequential':
                dc = 1.15 # spacing between colormaps
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap,
                           s=300, linewidths=0.)
                if i==2:
                    ax.axis([-0.1,4.1,0,100])
                else:
                    ax.axis([-0.1,4.7,0,100])
                locs.append(x[-1]+j*dc) # store locations for colormap labels

            elif cmap_category=='Sequential':
                dc = 0.6 # spacing between colormaps
                # These colormaps all start at high lightness but we want them
                # reversed to look nice in the plot, so reverse the order.
                ax.scatter(x+j*dc, lab[0,::-1,0], c=x[::-1], cmap=cmap,
                           s=300, linewidths=0.)
                if i==2:
                    ax.axis([-0.1,4.1,0,100])
                else:
                    ax.axis([-0.1,4.7,0,100])
                locs.append(x[-1]+j*dc) # store locations for colormap labels

            elif cmap_category=='Sequential (2)':
                dc = 1.15
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap,
                           s=300, linewidths=0.)
                ax.axis([-0.1,7.0,0,100])
                # store locations for colormap labels
                locs.append(x[-1]+j*dc)

            elif cmap_category=='Diverging':
                dc = 1.2
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap,
                           s=300, linewidths=0.)
                ax.axis([-0.1,7.1,0,100])
                # store locations for colormap labels
                locs.append(x[int(x.size/2.)]+j*dc)
            elif cmap_category=='Qualitative':
                dc = 1.3
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap,
                           s=300, linewidths=0.)
                ax.axis([-0.1,6.3,0,100])
                # store locations for colormap labels
                locs.append(x[int(x.size/2.)]+j*dc)

            elif cmap_category=='Miscellaneous':
                dc = 1.25
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap,
                           s=300, linewidths=0.)
                ax.axis([-0.1,6.1,0,100])
                # store locations for colormap labels
                locs.append(x[int(x.size/2.)]+j*dc)

            # Set up labels for colormaps
            ax.xaxis.set_ticks_position('top')
            ticker = mpl.ticker.FixedLocator(locs)
            ax.xaxis.set_major_locator(ticker)
            formatter = mpl.ticker.FixedFormatter(cmap_list[i*dsub:(i+1)*dsub])
            ax.xaxis.set_major_formatter(formatter)
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_rotation(60)

    ax.set_xlabel(cmap_category + ' colormaps', fontsize=14)
    fig.text(0.0, 0.55, 'Lightness $L^*$', fontsize=12,
             transform=fig.transFigure, rotation=90)

    fig.tight_layout(h_pad=0.05, pad=1.5)
    plt.show()
