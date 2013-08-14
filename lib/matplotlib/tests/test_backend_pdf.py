# -*- encoding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import six

import io

import numpy as np

from matplotlib import cm, rcParams
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
import os

# if 'TRAVIS' not in os.environ:
#     @image_comparison(baseline_images=['pdf_use14corefonts'], extensions=['pdf'])
#     def test_use14corefonts():
#         rcParams['pdf.use14corefonts'] = True
#         rcParams['font.family'] = 'sans-serif'
#         rcParams['font.size'] = 8
#         rcParams['font.sans-serif'] = ['Helvetica']

#         title = 'Test PDF backend with option use14corefonts=True'

#         text = '''A three-line text positioned just above a blue line
#         and containing some French characters and the euro symbol:
#         "Merci pépé pour les 10 €"'''

#         plt.figure()
#         plt.title(title)
#         plt.text(0.5, 0.5, text, horizontalalignment='center',
#                  verticalalignment='bottom',
#                  fontsize=24)
#         plt.axhline(0.5, linewidth=0.5)

@cleanup
def test_type42():
    rcParams['pdf.fonttype'] = 42

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    fig.savefig(io.BytesIO())
