# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         r"\usepackage{units}",         # load additional packages
         r"\usepackage{metalogo}",
         r"\usepackage{unicode-math}",  # unicode math setup
         r"\setmathfont{xits-math.otf}",
         r"\setmainfont{DejaVu Serif}", # serif font via preamble
         ]
}
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
plt.figure(figsize=(4.5,2.5))
plt.plot(range(5))
plt.xlabel(u"unicode text: я, ψ, €, ü, \\unitfrac[10]{°}{µm}")
plt.ylabel(u"\\XeLaTeX")
plt.legend([u"unicode math: $λ=∑_i^∞ μ_i^2$"])
plt.tight_layout(.5)

plt.savefig("pgf_preamble.pdf")
plt.savefig("pgf_preamble.png")
