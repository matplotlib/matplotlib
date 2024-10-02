
import matplotlib.pyplot as plt
import numpy as np

theta = np.arange(0, 2 * np.pi, np.pi / 8)
r = theta / np.pi / 2 + 0.5

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.errorbar(theta, r, xerr=0.1, yerr=0.1, capsize=7, fmt="o", c="seagreen")
ax.set_title("Pretty polar error bars")
plt.show()