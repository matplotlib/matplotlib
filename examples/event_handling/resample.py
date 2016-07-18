import numpy as np
import matplotlib.pyplot as plt


# A class that will downsample the data and recompute when zoomed.
class DataDisplayDownsampler(object):
    def __init__(self, xdata, ydata):
        self.origYData = ydata
        self.origXData = xdata
        self.ratio = 5
        self.delta = xdata[-1] - xdata[0]

    def downsample(self, xstart, xend):
        # Very simple downsampling that takes the points within the range
        # and picks every Nth point
        mask = (self.origXData > xstart) & (self.origXData < xend)
        xdata = self.origXData[mask]
        xdata = xdata[::self.ratio]

        ydata = self.origYData[mask]
        ydata = ydata[::self.ratio]

        return xdata, ydata

    def update(self, ax):
        # Update the line
        lims = ax.viewLim
        if np.abs(lims.width - self.delta) > 1e-8:
            self.delta = lims.width
            xstart, xend = lims.intervalx
            self.line.set_data(*self.downsample(xstart, xend))
            ax.figure.canvas.draw_idle()

# Create a signal
xdata = np.linspace(16, 365, 365-16)
ydata = np.sin(2*np.pi*xdata/153) + np.cos(2*np.pi*xdata/127)

d = DataDisplayDownsampler(xdata, ydata)

fig, ax = plt.subplots()

# Hook up the line
d.line, = ax.plot(xdata, ydata, 'o-')
ax.set_autoscale_on(False)  # Otherwise, infinite loop

# Connect for changing the view limits
ax.callbacks.connect('xlim_changed', d.update)

plt.show()
