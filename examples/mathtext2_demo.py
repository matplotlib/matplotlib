#!/usr/bin/env python
"""

In order to use mathtext2, you must build matplotlib.ft2font.  This is
built by default in the windows installer.

For other platforms, edit setup.py and set

BUILD_FT2FONT = True

You have to put the following lines in your matplotlibrc file if you want to
enable mathtext2 globaly (not needed for running this example):

mathtext.mathtext2: True            # Needed to enable the new mathtext
mathtext.rm     :   FreeSerif.ttf
mathtext.it     :   FreeSerifItalic.ttf     # Text italic
mathtext.tt     :   FreeMono.ttf    # Typewriter (monospaced)
mathtext.mit    :   FreeSerifItalic.ttf     # Math italic
mathtext.cal    :   FreeSansOblique.ttf # Caligraphic
mathtext.nonascii:  FreeSerif.ttf # #Used for \sum, \infty etc.

Note that "FreeSerif.ttf" etc. may be replaced by any font. Font files must be
in your system's font path.

Only the first parameter must be set (mathtext2 uses BaKoMa fonts by
default, and they come packaged with matplotlib, so the above lines
override them) because mathtext2 is disabled by default.

This demo assumes that you have FreeSerif.ttf installed.
You can get FreeSerif.ttf (and other files) from:
http://download.savannah.gnu.org/releases/freefont/
if you are on windows. On linux, they are usually shipped by default.
FreeFonts are distributed under GPL.

"""
# We override the default params
from matplotlib import rcParams
rcParams['mathtext.mathtext2'] = True

# You can put other fonts to override the default ones
rcParams['mathtext.rm'] = 'FreeSerif.ttf'
rcParams['mathtext.it'] = 'FreeSerifItalic.ttf'
rcParams['mathtext.tt'] = 'FreeMono.ttf'
rcParams['mathtext.mit'] = 'FreeSerifItalic.ttf'
rcParams['mathtext.cal'] = 'FreeSansOblique.ttf'

# This is used by mathtext2 to find chars with ord > 255 (Unicode characters)
rcParams['mathtext.nonascii'] = 'FreeSerif.ttf'
from pylab import *
subplot(111, axisbg='y')
plot([1,2,3], 'r')
x = arange(0.0, 3.0, 0.1)
grid(True)

tex = r'$1+1\ u_{x^2_1}^{y_{-q_u}}$'
#text(0.5, 2., tex, fontsize=20)
#show()
#xlabel(r'$\Delta_i^j$', fontsize=20)
#ylabel(r'$\Delta_{i+1}^j$', fontsize=20)
#tex = r'$\cal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\rm{sin}(2 \pi f x_i)$'
#tex = ur"$1^j_3$"
#tex = ur"$Tj_1j_jj_gT$"
#tex = ur"$F_1^1y_{1_{2_{3_2\sum_1^2{4}^6}}3}1_23$"
#tex = ur"$x_2{\cal TRALALA}\sum_1^2$"
#tex = ur"$a = x_2{\cal TRALALA}\sum_1^2$"
#tex = r'$K_{R osman dsfgs Tralala_{K_4^3}}X_1^1$'
#tex = ur"$Tutinjac\ fff\sin\exp$"
#tex = ur"$\sin\exp{\rm sin\ exp}$"
#tex = ur"$a^{\sin x}\sin b\sin(x/x), {\rm sin}(x/x){\rm sin\ }(x/x)$"
#tex = ur"$1\frac {\int_{-\infty}^\infty} 22$"
#tex = ur"$\rm a\vtext{Traktor}b$"
#tex = ur"$\frac{\int_{-\infty}^\infty} 2$"
#tex = ur"$1_\frac{\sum^2_{i_{23}=0}} 2678$"
#tex = ur"$1_{\frac{\sum^2_{i_{23}=0}}{\sum_{i=\frac94}^\infty} 345}678$"
text(0.5, 2., tex, fontsize=20)
tex = r'${\cal R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin\exp(2 \pi f x_i)$'
#text(1, 1.9, tex, fontsize=20)
tex = ur"$F_1^1y_{1_{2_{3_2\sum_1^2{4}^6}}3}1_23$"
#text(1, 1.7, tex, fontsize=20)
tex = ur"$x = \sin(\sum_{i=0}^\infty y_i)$"
#text(1, 1.5, tex, fontsize=20)
#title(r'$\Delta_i^j \hspace{0.4} \rm{versus} \hspace{0.4} \Delta_{i+1}^j$', fontsize=20)

savefig('mathtext_demo.png')
savefig('mathtext_demo.svg')
savefig('mathtext_demo.ps')


show()
