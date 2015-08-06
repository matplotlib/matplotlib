import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 4))
ax = plt.subplot(121)
ax.set_aspect(1)
plt.plot(np.arange(10))
plt.xlabel('this is a xlabel\n(with newlines!)')
plt.ylabel('this is vertical\ntest', multialignment='center')
plt.text(2, 7, 'this is\nyet another test',
         rotation=45,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')

plt.grid(True)

plt.subplot(122)

plt.text(0.29, 0.7, "Mat\nTTp\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

plt.text(0.34, 0.7, "Mag\nTTT\n123", size=18,
         va="baseline", ha="left", multialignment="left",
         bbox=dict(fc="none"))

plt.text(0.95, 0.7, "Mag\nTTT$^{A^A}$\n123", size=18,
         va="baseline", ha="right", multialignment="left",
         bbox=dict(fc="none"))

plt.xticks([0.2, 0.4, 0.6, 0.8, 1.],
           ["Jan\n2009", "Feb\n2009", "Mar\n2009", "Apr\n2009", "May\n2009"])

plt.axhline(0.7)
plt.title("test line spacing for multiline text")

plt.subplots_adjust(bottom=0.25, top=0.8)
plt.draw()
plt.show()
