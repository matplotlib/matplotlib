"""
=================================
Rendering math equation using TeX
=================================

You can use TeX to render all of your Matplotlib text by setting
:rc:`text.usetex` to True.  This requires that you have TeX and the other
dependencies described in the :doc:`/tutorials/text/usetex` tutorial properly
installed on your system.  Matplotlib caches processed TeX expressions, so that
only the first occurrence of an expression triggers a TeX compilation. Later
occurrences reuse the rendered image from the cache and are thus faster.

Unicode input is supported, e.g. for the y-axis label in this example.
"""

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(t, s)

ax.set_xlabel(r'\textbf{time (s)}')
ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
plt.show()
