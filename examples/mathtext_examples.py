#!/usr/bin/env python

import os, sys, re

import gc

stests = [
    r'Kerning: AVA $AVA$',
    r'\$100.00 $\alpha \_$',
    r'$\frac{\$100.00}{y}$',
    r'$x   y$',
    r'$x+y\ x=y\ x<y\ x:y\ x,y\ x@y$',
    r'$100\%y\ x*y\ x/y x\$y$',
    r'$x\leftarrow y\ x\forall y\ x-y$',
    r'$x \sf x \bf x {\cal X} \rm x$',
    r'$x\ x\,x\;x\quad x\qquad x\!x\hspace{ 0.5 }y$',
    r'$\{ \rm braces \}$',
    r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$',
    r'$\left(x\right)$',
    r'$\sin(x)$',
    r'$x_2$',
    r'$x^2$',
    r'$x^2_y$',
    r'$x_y^2$',
    r'$\prod_{i=\alpha_{i+1}}^\infty$',
    r'$x = \frac{x+\frac{5}{2}}{\frac{y+3}{8}}$',
    r'$dz/dt = \gamma x^2 + {\rm sin}(2\pi y+\phi)$',
    r'Foo: $\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau}$',
    r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i \sin(2 \pi f x_i)$',
#    r'$\bigodot \bigoplus {\sf R} a_i{\rm sin}(2 \pi f x_i)$',
    r'Variable $i$ is good',
    r'$\Delta_i^j$',
    r'$\Delta^j_{i+1}$',
    r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{\imath}\tilde{n}\vec{q}$',
    r'$_i$',
    r"$\arccos((x^i))$",
    r"$\gamma = \frac{x=\frac{6}{8}}{y} \delta$",
    r'$\limsup_{x\to\infty}$',
    r'$\oint^\infty_0$',
    r"$f^'$",
    r'$\frac{x_2888}{y}$',
    r"$\sqrt[3]{\frac{X_2}{Y}}=5$",
    r"$\sqrt[5x\pi]{\prod^\frac{x}{2\pi^2}_\infty}$",
    r"$\sqrt[3]{x}=5$",
    r'$\frac{X}{\frac{X}{Y}}$',
    # From UTR #25
    r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",
    r'$\mathcal{H} = \int d \tau \left(\epsilon E^2 + \mu H^2\right)$',
    r'$\widehat{abc}\widetilde{def}$',
    r'$\Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega$',
    r'$\alpha \beta \gamma \delta \epsilon \zeta \eta \theta \iota \lambda \mu \nu \xi \pi \kappa \rho \sigma \tau \upsilon \phi \chi \psi$',
    ur'Generic symbol: $\u23ce \mathrm{\ue0f2 \U0001D538}$'
    ]

from pylab import *

def doall():
    tests = stests
    
    figure(figsize=(8, (len(tests) * 1) + 2))
    plot([0, 0], 'r')
    grid(False)
    axis([0, 3, -len(tests), 0])
    yticks(arange(len(tests)) * -1)
    for i, s in enumerate(tests):
        print (i, s)
        text(0.1, -i, s, fontsize=20)

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
