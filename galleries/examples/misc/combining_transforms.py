"""
=========================
Combining Transformations
=========================

This example showcases how to combine transformations, including Affine
and blended transformations.

The goal of this exercise is to plot some circles with correct aspect
ratios on axes that are unequal. The interesting thing is that the
circles have a fixed size along the y-axis in data space rather than
axes, figure or display space.

To ensure a large disparity in the data axis scales, we plot
categorical vs time data. The x-axis is set to a short time interval of
a few minutes. The y-axis is integers corresponding to the category
index.

The transform graph is constructed as follows:

- A blended transform is made to scale the data:

  - The y direction is the scale-only portion of ``ax.transData``
    obtained with ``AffineDeltaTransform``
  - The x direction is a reflection of the y, made using an ``Affine2D``
    reflection matrix

- The blended transformation is added to a ``ScaledTranslation`` to place
  the circles correctly

As a secondary showcase, this example shows how to work with datetimes
along one axis when constructing the rectangular patches behind the
circles. It also demonstrates a way to increment the line color cycler.
"""

from datetime import datetime

import numpy as np

from matplotlib import dates as mdate
from matplotlib import patches as mpatch
from matplotlib import pyplot as plt
from matplotlib import transforms as mtrans

data = {
    "A": [datetime(2024, 4, 10, 3, 10, 22),
          datetime(2024, 4, 10, 3, 21, 13),
          datetime(2024, 4, 10, 3, 25, 41)],
    "B": [datetime(2024, 4, 10, 3, 15, 55),
          datetime(2024, 4, 10, 3, 40, 8)],
    "C": [datetime(2024, 4, 10, 3, 12, 18),
          datetime(2024, 4, 10, 3, 23, 32),
          datetime(2024, 4, 10, 3, 32, 12)],
}

fig, ax = plt.subplots(constrained_layout=True)
ax.invert_yaxis()

# The scaling transforms only need to be set up once
vertical_scale_transform = mtrans.AffineDeltaTransform(ax.transData)
reflection = mtrans.Affine2D.from_values(0, 1, 1, 0, 0, 0)
uniform_scale_transform = mtrans.blended_transform_factory(
    reflection + vertical_scale_transform + reflection, vertical_scale_transform)

# Draw some rectangle spanning each dataset
for i, dates in enumerate(data.values()):
    start = mdate.date2num(dates[0])
    end = mdate.date2num(dates[-1])
    width = end - start
    color = ax._get_lines.get_next_color()
    ax.add_patch(mpatch.Rectangle((start, i - 0.4), width, 0.8, color=color))

    # Draw a circle at each event
    for event in dates:
        x = mdate.date2num(event)
        t = uniform_scale_transform + mtrans.ScaledTranslation(x, i, ax.transData)
        ax.add_patch(mpatch.Circle((0, 0), 0.2, facecolor="w", edgecolor="k",
                                   linewidth=2, transform=t))

# Set the y-axis to show the data labels
ax.set_yticks(np.arange(len(data)))
ax.set_yticklabels(list(data))

# Set the x-axis to display datetimes
ax.xaxis.set_major_locator(locator := mdate.AutoDateLocator())
ax.xaxis.set_major_formatter(mdate.AutoDateFormatter(locator))

ax.autoscale_view()

plt.show()
