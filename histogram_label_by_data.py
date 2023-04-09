import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(0, 1, 2000)

fig, ax = plt.subplots()

ax.hist(data, bins=4)

names = ['lowest', 'low', 'high', 'highest']

ax.label_by_line(names)

plt.show()
