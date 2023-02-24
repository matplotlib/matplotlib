"""
=========
PGF fonts
=========
"""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    # Use LaTeX default serif font.
    "font.serif": [],
    # Use specific cursive fonts.
    "font.cursive": ["Comic Neue", "Comic Sans MS"],
})

fig, ax = plt.subplots(figsize=(4.5, 2.5))

ax.plot(range(5))

ax.text(0.5, 3., "serif")
ax.text(0.5, 2., "monospace", family="monospace")
ax.text(2.5, 2., "sans-serif", family="DejaVu Sans")  # Use specific sans font.
ax.text(2.5, 1., "comic", family="cursive")
ax.set_xlabel("µ is not $\\mu$")

fig.tight_layout(pad=.5)

fig.savefig("pgf_fonts.pdf")
fig.savefig("pgf_fonts.png")
