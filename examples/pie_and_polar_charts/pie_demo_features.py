"""
Demo of a basic pie chart plus a few additional features.

In addition to the basic pie chart, this demo shows a few optional features:

    * slice labels
    * auto-labeling the percentage
    * offsetting a slice with "explode"
    * drop-shadow
    * custom start angle

Note about the custom start angle:

The default ``startangle`` is 0, which would start the "Frogs" slice on the
positive x-axis. This example sets ``startangle = 90`` such that everything is
rotated counter-clockwise by 90 degrees, and the frog slice starts on the
positive y-axis.
"""
import matplotlib.pyplot as plt


def pie_demo_features(ax, sizes, labels, colors, explode):
    """
    produces a simple pie plot on ax according to sizes, labels, colors and explode.

    Parameters
    ----------
    ax :  PolarAxesSubplot
          Axes on which to plot polar_bar

    sizes : array
            comparative sizes of pie slices

    labels : array
           names of pie slices

    colors : array
           colors of pie slices

    explode : tuple
           how far to explode each slice (0 for don't explode)

    Returns
    -------
    pi : artist object returned

         Returns artist for further modification.
    """

    pi = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect rtio to be equal so that pie is drawn as a circle.
    ax.axis('equal')
    return pi

# Example data:
# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax = plt.subplot(111)

pie_demo_features(ax, sizes, labels, colors, explode)
plt.show()
