'''
For each colormap, plot the lightness parameter L* from CIELAB colorspace along the y axis vs index through the colormap. Colormaps are examined in categories as in the original matplotlib gallery of colormaps.
'''

from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 14})
mpl.rcParams['font.sans-serif'] = 'Arev Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.cal'] = 'cursive'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.tt'] = 'monospace'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.bf'] = 'sans:bold'
mpl.rcParams['mathtext.sf'] = 'sans'
mpl.rcParams['mathtext.fallback_to_cm'] = 'True'

# Have colormaps separated into categories: http://matplotlib.org/examples/color/colormaps_reference.html

cmaps = [('Sequential',     ['binary', 'Blues', 'BuGn', 'BuPu', 'gist_yarg',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential2', ['afmhot', 'autumn', 'bone', 'cool', 'copper',
                             'gist_gray', 'gist_heat', 'gray', 'hot', 'pink',
                             'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'hsv', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3', 'spectral']),
         ('Miscellaneous',  ['gist_earth', 'gist_ncar', 'gist_rainbow',
                             'gist_stern', 'jet', 'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'ocean', 'rainbow',
                             'terrain', 'flag', 'prism'])]

# indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
for cmap_category, cmap_list in cmaps:

    # Do subplots so that colormaps have enough space. 5 per subplot?
    dsub = 5 # number of colormaps per subplot
    if cmap_category == 'Diverging': # because has 13 colormaps
        dsub = 6
    elif cmap_category == 'Sequential2':
        dsub = 7
    elif cmap_category == 'Sequential':
        dsub = 7
    nsubplots = int(np.ceil(len(cmap_list)/float(dsub)))

    fig = plt.figure(figsize=(11.5,4*nsubplots))

    for i, subplot in enumerate(xrange(nsubplots)):

        locs = [] # locations for text labels

        ax = fig.add_subplot(nsubplots, 1, i+1)
        # pdb.set_trace()

        for j, cmap in enumerate(cmap_list[i*dsub:(i+1)*dsub]):

            # Get rgb values for colormap
            rgb = cm.get_cmap(cmap)(x)[np.newaxis,:,:3]

            # Get colormap in CIE LAB. We want the L here.
            lab = color.rgb2lab(rgb)

            # Plot colormap L values
            # Do separately for each category so each plot can be pretty
            # to make scatter markers change color along plot: http://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
            if cmap_category=='Sequential':
                dc = 0.6 # spacing between colormaps
                ax.scatter(x+j*dc, lab[0,::-1,0], c=x, cmap=cmap + '_r', s=300, linewidths=0.)
                if i==2:
                    ax.axis([-0.1,4.1,0,100])
                else:
                    ax.axis([-0.1,4.7,0,100])
                locs.append(x[-1]+j*dc) # store locations for colormap labels

            elif cmap_category=='Sequential2':
                dc = 1.15
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap, s=300, linewidths=0.)
                if i==0:
                    ax.axis([-0.1,8.1,0,100])
                else:
                    ax.axis([-0.1,7.0,0,100])
                locs.append(x[-1]+j*dc) # store locations for colormap labels

            elif cmap_category=='Diverging':
                dc = 1.2
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap, s=300, linewidths=0.)
                if i==0:
                    ax.axis([-0.1,7.1,0,100])
                else:
                    ax.axis([-0.1,6,0,100])
                locs.append(x[int(x.size/2.)]+j*dc) # store locations for colormap labels

            elif cmap_category=='Qualitative':
                dc = 1.3
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap, s=300, linewidths=0.)
                ax.axis([-0.1,6.3,0,100])
                locs.append(x[int(x.size/2.)]+j*dc) # store locations for colormap labels

            elif cmap_category=='Miscellaneous':
                dc = 1.25
                ax.scatter(x+j*dc, lab[0,:,0], c=x, cmap=cmap, s=300, linewidths=0.)
                ax.axis([-0.1,6.1,0,100])
                locs.append(x[int(x.size/2.)]+j*dc) # store locations for colormap labels

            # Set up labels for colormaps
            ax.xaxis.set_ticks_position('top')
            ticker = mpl.ticker.FixedLocator(locs)
            ax.xaxis.set_major_locator(ticker)
            formatter = mpl.ticker.FixedFormatter(cmap_list[i*dsub:(i+1)*dsub])
            ax.xaxis.set_major_formatter(formatter)
            labels = ax.get_xticklabels()
            for label in labels:
                label.set_rotation(60)

    ax.set_xlabel(cmap_category + ' colormaps', fontsize=22)
    fig.text(-0.005, 0.55, 'Lightness $L^*$', fontsize=18, transform=fig.transFigure, rotation=90)

    fig.tight_layout(h_pad=0.05)
    plt.show
