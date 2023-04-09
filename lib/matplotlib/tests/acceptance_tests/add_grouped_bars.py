import matplotlib.pyplot as plt
import numpy as np
heights = {'River Depth': (8.9, 12.3, 1.2), 'River Length': (
    40.0, 47.0, 55.0), 'River Volume': (56.34, 15.2, 12.29)}
rivers = ('Mackenzie River', 'Saskatchewan River', 'Fraser River')
width = 0.25
x = np.arange(len(rivers))
fig, ax = plt.subplots()
ax.grouped_bar(x, heights, width, rivers)
ax.spines['top'].set_visible(False)
plt.show()
