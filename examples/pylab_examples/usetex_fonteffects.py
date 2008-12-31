# This script demonstrates that font effects specified in your pdftex.map
# are now supported in pdf usetex.

import matplotlib
matplotlib.rc('text', usetex=True)
import pylab

def setfont(font):
    return r'\font\a %s at 14pt\a ' % font

for y, font, text in zip(range(5),
                         ['ptmr8r', 'ptmri8r', 'ptmro8r', 'ptmr8rn', 'ptmrr8re'],
                         ['Nimbus Roman No9 L ' + x for x in
                          ['', 'Italics (real italics for comparison)',
                           '(slanted)', '(condensed)', '(extended)']]):
    pylab.text(0, y, setfont(font) + text)

pylab.ylim(-1, 5)
pylab.xlim(-0.2, 0.6)
pylab.setp(pylab.gca(), frame_on=False, xticks=(), yticks=())
pylab.title('Usetex font effects')
pylab.savefig('usetex_fonteffects.pdf')
