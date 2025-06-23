import numpy as np
import matplotlib.pyplot as plt

# Lemniscate of Bernoulli in polar coordinates: r^2 = 2 * cos(2θ)
theta = np.linspace(-np.pi/4, np.pi/4, 1000)  # restrict to real r values where cos(2θ)>0
r = np.sqrt(2 * np.cos(2 * theta))

# For the full lemniscate, mirror the positive r with negative r for the other lobes
theta_full = np.concatenate([theta, theta + np.pi])
r_full = np.concatenate([r, -r])

plt.figure(figsize=(10, 5))

# Subplot 1: with ticks
ax1 = plt.subplot(121, projection='polar')
ax1.plot(theta_full, r_full, label='Lemniscate of Bernoulli')
ax1.set_title("Lemniscate with Ticks")
ax1.legend()

# Subplot 2: without ticks
ax2 = plt.subplot(122, projection='polar')
ax2.plot(theta_full, r_full, label='Lemniscate of Bernoulli')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Lemniscate without Ticks")
ax2.legend()

plt.show()
