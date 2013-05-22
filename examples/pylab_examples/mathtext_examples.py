#!/usr/bin/env python

from __future__ import print_function

import os, sys, re

import gc

stests = [
    r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$',
    r"$\gamma = \frac{x=\frac{6}{8}}{y} \delta$",
    r'$\limsup_{x\to\infty}$',
    r'$\oint^\infty_0$',
    r"$\sqrt[5]{\prod^\frac{x}{2\pi^2}_\infty}$",
    # From UTR #25
    r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",
    r'$\mathcal{H} = \int d \tau \left(\epsilon E^2 + \mu H^2\right)$',
    r'$\widehat{abc}\widetilde{def}$',
    #ur'Generic symbol: $\u23ce$',
   ]

#if sys.maxunicode > 0xffff:
#    stests.append(ur'$\mathrm{\ue0f2 \U0001D538}$')


from pylab import *

def doall():
    tests = stests

    figure(figsize=(8, (len(tests) * 1.0) + 2), facecolor='w')
    for i, s in enumerate(tests):
        print (i, s)
        figtext(0.1, float(i + 1) / (len(tests) + 2), s, fontsize=20)

    savefig('mathtext_examples')
    #close('all')
    show()

if '--latex' in sys.argv:
    fd = open("mathtext_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i, s in enumerate(stests):
        s = re.sub(r"(?<!\\)\$", "$$", s)
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex mathtext_examples.ltx")
else:
    doall()
