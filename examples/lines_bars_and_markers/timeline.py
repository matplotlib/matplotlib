"""
===============================================
Creating a timeline with lines, dates, and text
===============================================

How to create a simple timeline using Matplotlib release dates.

Timelines can be created with a collection of dates and text. In this example,
we show how to create a simple timeline using the dates for recent releases
of Matplotlib. First, we'll pull the data from GitHub.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import urllib.request
import json

# Grab a list of Matplotlib releases
url = 'https://api.github.com/repos/matplotlib/matplotlib/releases'
data = json.loads(urllib.request.urlopen(url).read().decode())

names = []
dates = []
for irelease in data:
    if 'rc' not in irelease['tag_name'] and 'b' not in irelease['tag_name']:
        names.append(irelease['tag_name'])
        # Convert date strings (e.g. 2014-10-18T18:56:23Z) to datetime
        dates.append(datetime.strptime(irelease['published_at'],
                     "%Y-%m-%dT%H:%M:%SZ"))

##############################################################################
# Next, we'll iterate through each date and plot it on a horizontal line.
# We'll add some styling to the text so that overlaps aren't as strong.
#
# Note that Matplotlib will automatically plot datetime inputs.

levels = np.array([-5, 5, -3, 3, -1, 1])
fig, ax = plt.subplots(figsize=(20, 5))

# Create the base line
start = min(dates)
stop = max(dates)
ax.plot((start, stop), (0, 0), 'k', alpha=.5)

# Iterate through releases annotating each one
for ii, (iname, idate) in enumerate(zip(names, dates)):
    level = levels[ii % 6]
    vert = 'top' if level < 0 else 'bottom'

    ax.scatter(idate, 0, s=100, facecolor='w', edgecolor='k', zorder=9999)
    # Plot a line up to the text
    ax.plot((idate, idate), (0, level), c='r', alpha=.7)
    # Give the text a faint background and align it properly
    ax.text(idate, level, iname,
            horizontalalignment='right', verticalalignment=vert, fontsize=14,
            backgroundcolor=(1., 1., 1., .3))
ax.set(title="Matplotlib release dates")
# Set the xticks formatting
# format xaxis with 3 month intervals
ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=3))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate()

# Remove components for a cleaner look
plt.setp((ax.get_yticklabels() + ax.get_yticklines() +
          list(ax.spines.values())), visible=False)
plt.show()
