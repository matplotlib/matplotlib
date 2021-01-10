"""
======
Slider
======

Using the slider widget to control visual properties of your plot.

In this example, sliders are used to control the frequency and amplitude of
a sine wave. You can control many continuously-varying properties of your plot
in this way.

For a more detailed example of value snapping see
:doc:`/gallery/widgets/slider_snap_demo`.

For an example of using a `matplotlib.widgets.RangeSlider` to define a range
of values see :doc:`/gallery/widgets/range_slider`.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.arange(0.0, 1.0, 0.001)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the `~.Line2D` that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
# This slider will snap to discrete values as defind by ``valstep``.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Frequency',
    valmin=0.1,
    valmax=30.0,
    valinit=init_amplitude,
    valstep=5.0
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0.1,
    valmax=10.0,
    valinit=init_amplitude,
    orientation="vertical"
)


def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.widgets.Button
matplotlib.widgets.Slider
