import numpy as np
import matplotlib.pyplot as plt
from scikits.audiolab import wavread


# A class that will downsample the data and recompute when zoomed.
class DataDisplayDownsampler(object):
    def __init__(self, xdata, ydata):
        self.origYData = ydata
        self.origXData = xdata
        self.numpts = 3000
        self.delta = xdata[-1] - xdata[0]

    def resample(self, xstart, xend):
        # Very simple downsampling that takes the points within the range
        # and picks every Nth point
        mask = (self.origXData > xstart) & (self.origXData < xend)
        xdata = self.origXData[mask]
        ratio = int(xdata.size / self.numpts) + 1
        xdata = xdata[::ratio]

        ydata = self.origYData[mask]
        ydata = ydata[::ratio]

        return xdata, ydata

    def update(self, ax):
        # Update the line
        lims = ax.viewLim
        if np.abs(lims.width - self.delta) > 1e-8:
            self.delta = lims.width
            xstart, xend = lims.intervalx
            self.line.set_data(*self.downsample(xstart, xend))
            ax.figure.canvas.draw_idle()

# Read data
data = wavread('/usr/share/sounds/purple/receive.wav')[0]
ydata = np.tile(data[:, 0], 100)
xdata = np.arange(ydata.size)

d = DataDisplayDownsampler(xdata, ydata)

fig, ax = plt.subplots()

# Hook up the line
xdata, ydata = d.downsample(xdata[0], xdata[-1])
d.line, = ax.plot(xdata, ydata)
ax.set_autoscale_on(False)  # Otherwise, infinite loop

# Connect for changing the view limits
ax.callbacks.connect('xlim_changed', d.update)

plt.show()
