# encoding: utf-8

import matplotlib
matplotlib.use('PDF')

from matplotlib import rcParams
import pylab

rcParams['pdf.use14corefonts'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 8
rcParams['font.sans-serif'] = ['Helvetica']

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
