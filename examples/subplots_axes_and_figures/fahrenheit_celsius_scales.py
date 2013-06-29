"""
Demo of how to display two scales on the left and right y axis.

This example uses the Fahrenheit and Celsius scales.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots()  # ax1 is the Fahrenheit scale
ax2 = ax1.twinx()          # ax2 is the Celsius scale

def fahrenheit2celsius(temp):
    """
    Returns temperature in Celsius.
    """
    return (5./9.)*(temp - 32)

def convert_ax2_to_celsius(ax1):
    """
    Update second axis according with first axis.
    """
    y1, y2 = ax1.get_ylim()
    ax2.set_ylim(fahrenheit2celsius(y1), fahrenheit2celsius(y2))
    ax2.figure.canvas.draw()

# automatically update ylim of ax2 when ylim of ax1 changes.
ax1.callbacks.connect("ylim_changed", convert_ax2_to_celsius)
ax1.plot(np.linspace(-40, 120, 100))
ax1.set_xlim(0, 100)

ax1.set_title('Two scales: Fahrenheit and Celsius')
ax1.set_ylabel('Fahrenheit')
ax2.set_ylabel('Celsius')

plt.show()
