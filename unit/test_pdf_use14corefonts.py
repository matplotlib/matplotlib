# -*- encoding: utf-8 -*-
"""
Test the PDF backend with the option use14corefonts=True.

Font cache issue
----------------

The font cache doesn't record whether it was build with
pdf.use14corefonts enabled or not, and the font name "Helvetica"
happens to match "Helvetica Narrow", whose metrics are included with
matplotlib, and using that AFM file without including the font itself
breaks the output.

As a workaround, please reset the font cache by deleting
~/.matplotlib/fontList.cache each time you enable or disable
use14corefonts.
"""

from matplotlib import rcParams

rcParams['backend'] = 'pdf'
rcParams['pdf.use14corefonts'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 8
rcParams['font.sans-serif'] = ['Helvetica']

import pylab

title = u'Test PDF backend with option use14corefonts=True'

text = u'''A three-line text positioned just above a blue line
and containing some French characters and the euro symbol:
"Merci pépé pour les 10 €"'''

pylab.figure(figsize=(6, 4))
pylab.title(title)
pylab.text(0.5, 0.5, text, horizontalalignment='center')
pylab.axhline(0.5, linewidth=0.5)
pylab.savefig('test_pdf_use14corefonts.pdf')
pylab.close()
