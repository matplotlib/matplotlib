# -----------------------------------------------------------------------------
# Animated display of last earthquakes (last 30 days)
# Author: Nicolas P. Rougier
#
# Based on : https://peak5390.wordpress.com
# -> 2012/12/08/matplotlib-basemap-tutorial-plotting-global-earthquake-activity/
# -----------------------------------------------------------------------------
import urllib
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from  matplotlib.animation import FuncAnimation


# Open the earthquake data
# -------------------------
# -> http://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php

feed = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"

# Significant earthquakes in the past 30 days
# url = urllib.urlopen(feed + "significant_month.csv")

# Earthquakes of magnitude > 4.5 in the past 30 days
url = urllib.urlopen(feed + "4.5_month.csv")

# Earthquakes of magnitude > 2.5 in the past 30 days
# url = urllib.urlopen(feed + "2.5_month.csv")

# Earthquakes of magnitude > 1.0 in the past 30 days
# url = urllib.urlopen(feed + "1.0_month.csv")


# Store earthquake data
# ---------------------
data = url.read().split('\n')[+1:-1]
E = np.zeros(len(data), dtype=[('position',  float, 2),
                               ('magnitude', float, 1)])
for i in range(len(data)):
    row = data[i].split(',')
    E['position'][i] = float(row[2]),float(row[1])
    E['magnitude'][i] = float(row[4])


# Create a new figure
fig = plt.figure(figsize=(14,7))
ax = plt.subplot(1,1,1)

# 50 represent the number of simultaneous displayed earthquakes
P = np.zeros(50, dtype=[('position', float, 2),
                        ('size',     float, 1),
                        ('growth',   float, 1),
                        ('color',    float, 4)])

# Choose your projection
map = Basemap(projection='robin', lat_0=0, lon_0=-130)
# map = Basemap(projection='mill')

map.drawcoastlines(color='0.50', linewidth=0.25)
map.fillcontinents(color='0.95')

# Scatter plot is used for the animation
scat = ax.scatter(P['position'][:,0], P['position'][:,1], P['size'], lw=0.5,
                  edgecolors = P['color'], facecolors='None', animated=True)

def update(frame):
    current = frame % len(E)
    i = frame % len(P)

    # Make all colors more transparent
    P['color'][:,3] = np.maximum(0, P['color'][:,3] - 1.0/len(P))
    # Make all circles bigger
    P['size'] += P['growth']

    #  Use oldest circle to represent current earthquake
    P['position'][i] = map(*E['position'][current])
    P['size'][i]     = 5
    P['color'][i]    = 1,0,0,1
    P['growth'][i]   = np.exp(E['magnitude'][current]) * 0.1

    # Update scatter plots
    scat.set_edgecolors(P['color'])
    scat.set_facecolors(P['color']*(1,1,1,0.25))
    scat.set_sizes(P['size'])
    scat.set_offsets(P['position'])
    return scat,


animation = FuncAnimation(fig, update, interval=10, blit=True)
plt.show()
