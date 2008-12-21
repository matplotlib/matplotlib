"""
Hatching (pattern filled polygons) is supported currently on PS and PDF
backend only.  See the set_patch method in
http://matplotlib.sf.net/matplotlib.patches.html#Patch
for details
"""
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.annotate("Hatch is only supported in the PS and PDF backend", (1, 1),
             xytext=(0, 5),
             xycoords="axes fraction", textcoords="offset points", ha="center"
             )
ax1.bar(range(1,5), range(1,5), color='gray', ecolor='black', hatch="/")


ax2 = fig.add_subplot(122)
bars = ax2.bar(range(1,5), range(1,5), color='gray', ecolor='black')

patterns = ('/', '+', 'x', '\\')
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)

plt.show()
