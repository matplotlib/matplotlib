import matplotlib.pyplot as plt
import numpy as np

xs, ys = np.mgrid[-180:180:0.5, -90:90:0.5]

data = np.sin(np.deg2rad(xs) * 4) + np.sin(np.deg2rad(ys) * 2)

plt.figure(figsize=(18, 12))
plt.pcolormesh(xs, ys, data, alpha=0.6)

plt.show()
