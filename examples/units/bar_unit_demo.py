"""
=========================
Group barchart with units
=========================

This is the same example as
:doc:`the barchart</gallery/lines_bars_and_markers/barchart>` in
centimeters.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

import numpy as np
from basic_units import cm, inch
import matplotlib.pyplot as plt

N=5
bream_means = [26*cm, 27*cm, 28*cm, 29*cm, 29*cm]
bream_std = [2*cm, 3*cm, 1*cm, 1*cm, 4*cm]

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
ax.bar(ind, bream_means, width, bottom=0*cm, yerr=bream_std, 
       label='Bream')

parkki_means = (14*cm, 13*cm, 17*cm, 16*cm, 20*cm)
parkki_std = (1*cm, 2*cm, 4*cm, 3*cm, 2*cm)
ax.bar(ind + width, parkki_means, width, bottom=0*cm, yerr=parkki_std,
       label='Parkki')

ax.set_title('Scores by group and length of Bream and Parkki fish')
ax.set_xticks(ind + width / 2, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

ax.legend()
ax.yaxis.set_units(inch)
ax.autoscale_view()

plt.show()