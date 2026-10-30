"""
======
Slider
======

In this example, sliders are used to control the frequency and amplitude of
a sine wave.

This demonstrates extending the functionality of a FuncContainer, adding
interactive elements.

"""

import inspect

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.data_containers.containers import FuncContainer
from matplotlib.widgets import Button, Slider


class SliderContainer(FuncContainer):
    def __init__(self, xfuncs, /, **sliders):
        self._sliders = sliders
        for slider in sliders.values():
            slider.on_changed(
                lambda _, sld=slider: sld.ax.figure.canvas.draw_idle(),
            )

        def get_needed_keys(f, offset=1):
            return tuple(inspect.signature(f).parameters)[offset:]

        super().__init__(
            {
                k: (
                    s,
                    # this line binds the correct sliders to the functions
                    # and makes lambdas that match the API FuncContainer needs
                    lambda x, keys=get_needed_keys(f), f=f: f(
                        x, *(sliders[k].val for k in keys)
                    ),
                )
                for k, (s, f) in xfuncs.items()
            },
        )

    def _query_hash(self, graph, parent_coordinates):
        key = super()._query_hash(graph, parent_coordinates)
        # inject the slider values into the hashing logic
        return hash((key, tuple(s.val for s in self._sliders.values())))


# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(-7, 7)

ax.set_xlabel("Time [s]")

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25, right=0.75)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label="Frequency [Hz]",
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical",
)

# Make a vertically oriented slider to control the phase
axphase = fig.add_axes([0.85, 0.25, 0.0225, 0.63])
phase_slider = Slider(
    ax=axphase,
    label="Phase [rad]",
    valmin=-2 * np.pi,
    valmax=2 * np.pi,
    valinit=0,
    orientation="vertical",
)

# set up the data container
fc = SliderContainer(
    {
        # the x data does not need the sliders values
        "x": (("N",), lambda t: t),
        "y": (
            ("N",),
            # the y data needs all three sliders
            lambda t, amplitude, frequency, phase: amplitude
            * np.sin(2 * np.pi * frequency * t + phase),
        ),
    },
    # bind the sliders to the data container
    amplitude=amp_slider,
    frequency=freq_slider,
    phase=phase_slider,
)

line, *_ = ax.plot([], [])
line.set_container(fc)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")
button.on_clicked(
    lambda _: [sld.reset() for sld in (freq_slider, amp_slider, phase_slider)]
)

plt.show()
