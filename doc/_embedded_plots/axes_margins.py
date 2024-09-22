import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.linspace(0, 1, 33)
y = -np.sin(x * 2*np.pi)
ax.plot(x, y, 'o')
ax.margins(0.5, 0.2)
ax.set_title("margins(x=0.5, y=0.2)")

# fix the Axes limits so that the following helper drawings
# cannot change them further.
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())


def arrow(p1, p2, **props):
    ax.annotate("", p1, p2,
                arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, **props))


axmin, axmax = ax.get_xlim()
aymin, aymax = ax.get_ylim()
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

y0 = -0.8
ax.axvspan(axmin, xmin, color=("orange", 0.1))
ax.axvspan(xmax, axmax, color=("orange", 0.1))
arrow((xmin, y0), (xmax, y0), color="sienna")
arrow((xmax, y0), (axmax, y0), color="orange")
ax.text((xmax + axmax)/2, y0+0.05, "x margin\n* x data range",
        ha="center", va="bottom", color="orange")
ax.text(0.55, y0+0.1, "x data range", va="bottom", color="sienna")

x0 = 0.1
ax.axhspan(aymin, ymin, color=("tab:green", 0.1))
ax.axhspan(ymax, aymax, color=("tab:green", 0.1))
arrow((x0, ymin), (x0, ymax), color="darkgreen")
arrow((x0, ymax), (x0, aymax), color="tab:green")
ax.text(x0, (ymax + aymax) / 2, "  y margin * y data range",
        va="center", color="tab:green")
ax.text(x0, 0.5, "  y data range", color="darkgreen")
