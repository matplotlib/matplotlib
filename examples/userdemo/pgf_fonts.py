"""
=========
Pgf Fonts
=========

"""

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [],                    # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
})

fig, ax = plt.subplots(figsize=(4.5, 2.5))

ax.plot(range(5))

ax.text(0.5, 3., "serif")
ax.text(0.5, 2., "monospace", family="monospace")
ax.text(2.5, 2., "sans-serif", family="sans-serif")
ax.text(2.5, 1., "comic sans", family="Comic Sans MS")
ax.set_xlabel("µ is not $\\mu$")

fig.tight_layout(pad=.5)

fig.savefig("pgf_fonts.pdf")
fig.savefig("pgf_fonts.png")
