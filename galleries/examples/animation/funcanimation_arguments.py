"""
==================================
Parameterized animation function
==================================


Create an `.FuncAnimation` animation updating function that takes as input the frame,
objects being animated, and data set.
"""

import functools

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


def update(frame, line, text, decades, widgets_data):
    # frame (int): The current frame number.
    # line (Line2D): The line object to update.
    # text (Text): The text annotation object to update.
    # decades (numpy.ndarray): Array of decades.
    # widgets_data (numpy.ndarray): Array of widgets data.

    current_decade = decades[frame]
    current_widgets = int(widgets_data[frame])

    line.set_data(decades[:frame + 1], widgets_data[:frame + 1])
    text.set_text(f'Decade: {current_decade}\nNumber of Widgets: {current_widgets}')

    return line, text

# Set up the animation

# Constants
decades = np.arange(1940, 2020, 10)
initial_widgets = 10000  # Rough estimate of the no. of widgets in the 1950s

# Generate rough growth data
growth_rate = np.random.uniform(1.02, 3.10, size=len(decades))
widgets_data = np.cumprod(growth_rate) * initial_widgets

# Set up the initial plot
fig, ax = plt.subplots()

# create an empty line
line, = ax.plot([], [])

# display the current decade
text = ax.text(0.5, 0.85, '', transform=ax.transAxes,
               fontsize=12, ha='center', va='center')

ax.set(xlabel='Decade', ylabel='Number of Widgets',
         xlim=(1940, 2020),
         ylim=(0, max(widgets_data) + 100000))


ani = animation.FuncAnimation(
    fig,
    # bind arguments to the update function, leave only frame unbound
    functools.partial(update,  line=line, text=text,
                      decades=decades, widgets_data=widgets_data),
    frames=len(decades),
    interval=1000,
)

plt.show()
