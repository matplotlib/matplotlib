import matplotlib.pyplot as plt
import numpy as np

def fmt_r(r):
    # Custom formatting for radial coordinates (r)
    return f"R={r:.2f}"

def fmt_theta(theta, pos=None):
    # Custom formatting for angular coordinates (theta)
    return f"Theta={np.degrees(theta):.2f}°"

def millions(x):
    return '$%1.1fM' % (x*1e-6)

x = np.random.rand(20)
y = 1e7 * np.random.rand(20)

# Create a polar subplot using plt.subplot() and specify projection as 'polar'
ax = plt.subplot(111, projection='polar')

# Set the radial tick labels using the custom formatting
ax.set_yticklabels([fmt_r(label) for label in ax.get_yticks()])

# Set the angular tick labels using the custom formatting
ax.set_xticklabels([fmt_theta(label) for label in ax.get_xticks()])

# Plot the data
ax.plot(x, y, 'o')

plt.show()
