"""
==================
Usetex Fonteffects
==================

This script demonstrates that font effects specified in your pdftex.map
are now supported in usetex mode.
"""

import matplotlib.pyplot as plt


def setfont(font):
    return rf'\font\a {font} at 14pt\a '


fig = plt.figure()
for y, font, text in zip(
    range(5),
    ['ptmr8r', 'ptmri8r', 'ptmro8r', 'ptmr8rn', 'ptmrr8re'],
    [f'Nimbus Roman No9 L {x}'
     for x in ['', 'Italics (real italics for comparison)',
               '(slanted)', '(condensed)', '(extended)']],
):
    fig.text(.1, 1 - (y + 1) / 6, setfont(font) + text, usetex=True)

fig.suptitle('Usetex font effects')
# Would also work if saving to pdf.
plt.show()
