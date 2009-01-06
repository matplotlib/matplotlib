"""
Hatching (pattern filled polygons) is supported currently in the PS,
PDF, SVG and Agg backends only.
"""
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.annotate("Hatch is only supported in the PS, PDF, SVG and Agg backends", (1, 1),
             xytext=(0, 5),
             xycoords="axes fraction", textcoords="offset points", ha="center"
             )
ax1.bar(range(1,5), range(1,5), color='red', edgecolor='black', hatch="/")
ax1.bar(range(1,5), [6] * 4, bottom=range(1,5), color='blue', edgecolor='black', hatch='//')

ax2 = fig.add_subplot(122)
bars = ax2.bar(range(1,5), range(1,5), color='yellow', ecolor='black') + \
    ax2.bar(range(1, 5), [6] * 4, bottom=range(1,5), color='green', ecolor='black')

patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)

plt.show()
