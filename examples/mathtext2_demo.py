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
#xlabel(r'$\Delta_i^j$', fontsize=20)
#ylabel(r'$\Delta_{i+1}^j$', fontsize=20)
#tex = r'$\cal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\rm{sin}(2 \pi f x_i)$'
#tex = ur"$1^2_3$"
#tex = ur"$Tj_1j_jj_gT$"
#tex = ur"$F_1^1y_{1_{2_{3_2\sum_1^2{4}^6}}3}1_23$"
#tex = ur"$x_2{\cal TRALALA}\sum_1^2$"
#tex = ur"$a = x_2{\cal TRALALA}\sum_1^2$"
#tex = r'$K_{R osman dsfgs Tralala_{K_4^3}}X_1^1$'
#tex = ur"$Tutinjac\ fff\sin\exp$"
#tex = ur"$\sin\exp{\rm sin\ exp}$"
#tex = ur"$a^{\sin x}\sin b\sin(x/x), {\rm sin}(x/x){\rm sin\ }(x/x)$"
#tex = ur"$\frac 3 2$"
#tex = ur"$3 2$"
#text(0.5, 2., tex, fontsize=50)
tex = r'${\cal R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin\exp(2 \pi f x_i)$'
text(1, 1.9, tex, fontsize=20)
tex = ur"$F_1^1y_{1_{2_{3_2\sum_1^2{4}^6}}3}1_23$"
text(1, 1.7, tex, fontsize=20)
tex = ur"$x = \sin(\sum_{i=0}^\infty y_i)$"
text(1, 1.5, tex, fontsize=20)
#title(r'$\Delta_i^j \hspace{0.4} \rm{versus} \hspace{0.4} \Delta_{i+1}^j$', fontsize=20)
savefig('mathtext_demo.png')
#savefig('mathtext_demo.ps')


show()
