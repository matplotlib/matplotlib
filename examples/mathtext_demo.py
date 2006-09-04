#!/usr/bin/env python
"""
 
In order to use mathtext, you must build matplotlib.ft2font.  This is
built by default in the windows installer.

For other platforms, edit setup.py and set

BUILD_FT2FONT = True

"""
from pylab import *
subplot(111, axisbg='y')
plot([1,2,3], 'r')
x = arange(0.0, 3.0, 0.1)

grid(True)
xlabel(r'$\Delta_i^j$', fontsize=20)
ylabel(r'$\Delta_{i+1}^j$', fontsize=20)
tex = r'$\cal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\rm{sin}(2 \pi f x_i)$'
text(1, 1.6, tex, fontsize=20)

#title(r'$\Delta_i^j \hspace{0.4} \rm{versus} \hspace{0.4} \Delta_{i+1}^j$', fontsize=20)
savefig('mathtext_demo')



show()
