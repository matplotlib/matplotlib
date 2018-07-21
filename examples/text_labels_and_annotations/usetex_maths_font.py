"""
============================
TeX and maths font selection
============================

Setting the text font in TeX does not (by default) change the maths font.
It is recommended to use ``text.latex.preamble`` to set the font to ensure
that both text and maths use the desired font settings.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(0.0, 1.0, 100)
y = np.cos(4 * np.pi * x) + 2

with mpl.rc_context(rc={'text.usetex': True,
                        'font.family': 'sans-serif',
                        'font.sans-serif': 'DejaVu Sans'}):
    fig1, ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax1.plot(x, y)
    ax1.set_title(r"Using `font.family', `font.sans-serif' params:"
                  '\n'+r'$\displaystyle\sum_{n=1}^\infty'
                  r'\frac{-e^{i\pi}}{2^n}$!'
                  r'$\leftarrow$ serif font (also in tick labels).',
                  fontsize=20, color='r')
    ax1.tick_params(axis='both', labelsize=20)
    plt.savefig('usetex_maths_DejaVu-Sans.png')


with mpl.rc_context(rc={'text.usetex': True,
                        'text.latex.preamble': r'\usepackage{cmbright}'}):

    fig2, ax2 = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax2.plot(x, y)
    ax2.set_title(r'Using \rm{\textbackslash usepackage\{cmbright\}}:'
                  '\n'+r'$\displaystyle\sum_{n=1}^\infty'
                  r'\frac{-e^{i\pi}}{2^n}$!'
                  r'$\leftarrow$ sans-serif font (also in tick labels).',
                  fontsize=20, color='r')
    ax2.tick_params(axis='both', labelsize=20)
    plt.savefig('usetex_maths_cmbright.png')