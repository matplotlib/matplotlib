# -----------------------------------------------------------------------------
# Rain simulation
# Author: Nicolas P. Rougier
# -----------------------------------------------------------------------------
import matplotlib
import numpy as np
matplotlib.use('TkAgg') # Required on OSX for animation
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation


# Create new figure
fig = plt.figure(figsize=(7,7))
ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)

# Create rain data
P = np.zeros(50, dtype=[('position', float, 2),
                        ('size',     float, 1),
                        ('growth',   float, 1),
                        ('color',    float, 4)])

# Scatter plot is used to animate rain drops
scat = ax.scatter(P['position'][:,0], P['position'][:,1], P['size'], lw=0.5,
                  animated=True, edgecolors = P['color'], facecolors='none')
ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])


def update(frame):
    i = frame % len(P)

    # Make all colors more transparent
    P['color'][:,3] -= 1.0/len(P)
    P['color'][:,3] = np.clip(P['color'][:,3], 0, 1)

    # Make all circles bigger
    P['size'] += P['growth']

    # Pick a new position for oldest rain drop
    P['position'][i] = np.random.uniform(0, 1, 2)

    # Reset size
    P['size'][i] = 5

    # Reset color
    P['color'][i] = (0, 0, 0, 1)

    # Choose a random growth factor
    P['growth'][i] = np.random.uniform(50, 200)

    # Update scatter plot
    scat.set_edgecolors(P['color'])
    scat.set_sizes(P['size'])
    scat.set_offsets(P['position'])

    return scat,

animation = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
