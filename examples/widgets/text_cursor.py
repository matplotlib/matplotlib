"""
======
Textcursor
======

"""
from matplotlib.widgets import TextCursor
import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, facecolor='#FFFFCC')

#x, y = 4*(np.random.rand(2, 100) - .5)
x = np.linspace(-2, 2, 1000)
#y = 2*np.sin(x*np.pi)
y=(x**2)-2
lin = ax.plot(x, y)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Set useblit=True on most backends for enhanced performance.
cursor = TextCursor(line=lin[0], numberformat="{0:.2f}\n{1:.2f}", dataaxis='x', offset=[10, 10], textprops={'color':'blue', 'fontweight':'bold'}, ax=ax, useblit=True, color='red', linewidth=2)

plt.show()