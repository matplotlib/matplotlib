#!/usr/bin/env python

from __future__ import unicode_literals

import os, sys, re

import gc

stests = [
    r'$\mathcircled{123} \mathrm{\mathcircled{123}} \mathbf{\mathcircled{123}}$',
    r'$\mathsf{Sans \Omega} \mathrm{\mathsf{Sans \Omega}} \mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard \pi}$',
    r'$\mathrm{\mathbb{Blackboard \pi}}$',
    r'$\mathbf{\mathbb{Blackboard \pi}}$',
    r'$\mathfrak{Fraktur} \mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$']

if sys.maxunicode > 0xffff:
    s = r'Direct Unicode: $\u23ce \mathrm{\ue0f2 \U0001D538}$'

from pylab import *

def doall():
    tests = stests

    figure(figsize=(8, (len(tests) * 1) + 2))
    plot([0, 0], 'r')
    grid(False)
    axis([0, 3, -len(tests), 0])
    yticks(arange(len(tests)) * -1)
    for i, s in enumerate(tests):
        #print (i, s.encode("ascii", "backslashreplace"))
        text(0.1, -i, s, fontsize=32)

    savefig('stix_fonts_example')
    #close('all')
    show()

if '--latex' in sys.argv:
    fd = open("stix_fonts_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i, s in enumerate(stests):
        s = re.sub(r"(?<!\\)\$", "$$", s)
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex stix_fonts_examples.ltx")
else:
    doall()
