# -----------------------------------------------------------------------------
# Rain simulation
# Author: Nicolas P. Rougier
# -----------------------------------------------------------------------------
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation


fig = plt.figure(figsize=(8,8))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)

P = np.zeros(50, dtype=[('position', float, 2),
                        ('size',     float, 1),
                        ('growth',   float, 1),
                        ('color',    float, 4)])
scat = ax.scatter(P['position'][:,0], P['position'][:,1], P['size'], lw=0.5,
                  animated=True, edgecolors = P['color'], facecolors='None')
ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])


def update(frame):
    i = frame % 50

    # Make all colors more transparent
    P['color'][:,3] = np.maximum(0, P['color'][:,3] - 1.0/len(P))
    # Make all circles bigger
    P['size'] += P['growth']

    #  Pick new position for oldest rain drop
    P['position'][i] = np.random.uniform(0,1,2)
    P['size'][i]     = 5
    P['color'][i]    = 0,0,0,1
    P['growth'][i]   = np.random.uniform(50,200)

    # Update scatter plots
    scat.set_edgecolors(P['color'])
    scat.set_sizes(P['size'])
    scat.set_offsets(P['position'])

    return scat,

animation = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
