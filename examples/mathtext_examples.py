#!/usr/bin/env python

import os, sys

stests = [
    r'Kerning: AVA $AVA$',
    r'$x   y$',
    r'$x+y\ x=y\ x<y\ x:y\ x,y\ x@y$',
    r'$100\%y\ x*y\ x/y x\$y$',
    r'$x\leftarrow y\ x\forall y$',
    r'$x \sf x \bf x {\cal X} \rm x$',
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
    r'$dz/dt \/ = \/ \gamma x^2 \/ + \/ {\rm sin}(2\pi y+\phi)$',
    r'Foo: $\alpha_{i+1}^j \/ = \/ {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau}$',
    r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i \sin(2 \pi f x_i)$',
#    r'$\bigodot \bigoplus {\sf R} a_i{\rm sin}(2 \pi f x_i)$',
    r'Variable $i$ is good',
    r'$\Delta_i^j$',
    r'$\Delta^j_{i+1}$',
    r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{\imath}\tilde{n}\vec{q}$',
    r'$_i$',
    r"$\arccos((x^i))$",
    r"$\gamma = \frac{x=\frac{6}{8}}{y} \delta$",
    r'$\"o\ddot o \'e\`e\~n\.x\^y$',
    
    ]

from pylab import *

if '--latex' in sys.argv:
    fd = open("mathtext_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i, s in enumerate(stests):
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex mathtext_examples.ltx")
else:
    for i, s in enumerate(stests):
        print "%02d: %s" % (i, s)
        plot([1,2,3], 'r')
        x = arange(0.0, 3.0, 0.1)

        grid(True)
        text(1, 1.6, s, fontsize=20)

        savefig('mathtext_example%02d' % i)
        figure()

