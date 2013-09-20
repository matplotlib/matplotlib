from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['table_zorder'],
                  extensions=['png'],
                  remove_text=True)
def test_zorder():
    data = [[66386, 174296],
            [58230, 381139]]

    colLabels = ('Freeze', 'Wind')
    rowLabels = ['%d year' % x for x in (100, 50)]

    cellText = []
    yoff = np.array([0.0] * len(colLabels))
    for row in reversed(data):
        yoff += row
        cellText.append(['%1.1f' % (x/1000.0) for x in yoff])

    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(t, np.cos(t), lw=4, zorder=2)

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='center',
              zorder=-2,
              )

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='upper center',
              zorder=4,
              )
    plt.yticks([])

@image_comparison(baseline_images=['table_row_label1', 'table_row_label2',
                                   'table_col_label1', 'table_col_label2',
                                   'table_plain.png'],
                  extensions=['png'])
def test_label_colours():
    dim = 3
    
    c = np.linspace(0, 1, dim)
    colours = plt.cm.RdYlGn(c)
    cellText = [['1'] * dim] * dim
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axis('off')
    ax1.table(cellText=cellText,
              rowColours=colours)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.axis('off')
    ax2.table(cellText=cellText,
              rowColours=colours,
              rowLabels=['Header'] * dim)

    fig = plt.figure()
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.axis('off')
    ax3.table(cellText=cellText,
              colColours=colours,
              loc='best')

    fig = plt.figure()
    ax4 = fig.add_subplot(1, 1, 1)
    ax4.axis('off')
    ax4.table(cellText=cellText,
              colColours=colours,
              colLabels=['Header'] * dim,
              loc='best')

    fig = plt.figure()
    ax5 = fig.add_subplot(1, 1, 1)
    ax5.axis('off')
    ax5.table(cellText=cellText,
              loc='center')
    
