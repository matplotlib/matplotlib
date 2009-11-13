# This is a demo of creating a pdf file with several pages.

import datetime
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *

# Create the PdfPages object to which we will save the pages:
pdf = PdfPages('multipage_pdf.pdf')

figure(figsize=(3,3))
plot(range(7), [3,1,4,1,5,9,2], 'r-o')
title('Page One')
savefig(pdf, format='pdf') # note the format='pdf' argument!
close()

rc('text', usetex=True)
figure(figsize=(8,6))
x = np.arange(0,5,0.1)
plot(x, np.sin(x), 'b-')
title('Page Two')
pdf.savefig() # here's another way - or you could do pdf.savefig(1)
close()

rc('text', usetex=False)
fig=figure(figsize=(4,5))
plot(x, x*x, 'ko')
title('Page Three')
pdf.savefig(fig) # or you can pass a Figure object to pdf.savefig
close()

# We can also set the file's metadata via the PdfPages object:
d = pdf.infodict()
d['Title'] = 'Multipage PDF Example'
d['Author'] = u'Jouni K. Sepp\xe4nen'
d['Subject'] = 'How to create a multipage pdf file and set its metadata'
d['Keywords'] = 'PdfPages multipage keywords author title subject'
d['CreationDate'] = datetime.datetime(2009,11,13)
d['ModDate'] = datetime.datetime.today()

# Remember to close the object - otherwise the file will not be usable
pdf.close()
