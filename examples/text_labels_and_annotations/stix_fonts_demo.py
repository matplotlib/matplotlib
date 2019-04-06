"""
===============
STIX Fonts Demo
===============

"""

import matplotlib.pyplot as plt
import numpy as np


circle123 = "\N{CIRCLED DIGIT ONE}\N{CIRCLED DIGIT TWO}\N{CIRCLED DIGIT THREE}"

tests = [
    r'$%s \mathrm{%s} \mathbf{%s}$' % ((circle123,) * 3),
    r'$\mathsf{Sans \Omega} \mathrm{\mathsf{Sans \Omega}}'
    r' \mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard \pi}$',
    r'$\mathrm{\mathbb{Blackboard \pi}}$',
    r'$\mathbf{\mathbb{Blackboard \pi}}$',
    r'$\mathfrak{Fraktur} \mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$',
]


fig = plt.figure(figsize=(8, (len(tests) * 1) + 2))
for i, s in enumerate(tests[::-1]):
    fig.text(0, (i + .5) / len(tests), s, fontsize=32)

plt.show()
