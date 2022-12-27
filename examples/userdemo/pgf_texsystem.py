"""
=============
PGF texsystem
=============
"""

import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

fig, ax = plt.subplots(figsize=(4.5, 2.5))

ax.plot(range(5))

ax.text(0.5, 3., "serif", family="serif")
ax.text(0.5, 2., "monospace", family="monospace")
ax.text(2.5, 2., "sans-serif", family="sans-serif")
ax.set_xlabel(r"µ is not $\mu$")

fig.tight_layout(pad=.5)

fig.savefig("pgf_texsystem.pdf")
fig.savefig("pgf_texsystem.png")
