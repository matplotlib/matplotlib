import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, ax = plt.subplots()

# Plot function
fp = ax.fplot(np.tan, [0, 2])
ax.set_xlim([1, 2])
plt.show()
