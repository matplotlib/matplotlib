import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfFile
from pylab import *

pdf = PdfFile('multipage_pdf.pdf')

figure(figsize=(3,3))
plot(range(7), [3,1,4,1,5,9,2], 'r-o')
title('Page One')
savefig(pdf, format='pdf')
close()

rc('text', usetex=True)
figure(figsize=(8,6))
x = np.arange(0,5,0.1)
plot(x, np.sin(x), 'b-')
title('Page Two')
savefig(pdf, format='pdf')
close()

rc('text', usetex=False)
figure(figsize=(4,5))
plot(x, x*x, 'ko')
title('Page Three')
savefig(pdf, format='pdf')
close()

pdf.close()
