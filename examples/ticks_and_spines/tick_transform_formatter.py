"""
Demo of the `matplotlib.ticker.TransformFormatter` class.

This code demonstrates two features:

  1. A linear transformation of the input values. A callable class for
     doing the transformation is presented as a recipe here. The data
     type of the inputs does not change.
  2. A transformation of the input type. The example here allows
     `matplotlib.ticker.StrMethodFormatter` to handle integer formats
     ('b', 'o', 'd', 'n', 'x', 'X'), which will normally raise an error
     if used directly. This transformation is associated with a
     `matplotlib.ticker.MaxNLocator` which has `integer` set to True to
     ensure that the inputs are indeed integers.

The same histogram is plotted in two sub-plots with a shared x-axis.
Each axis shows a different temperature scale: one in degrees Celsius,
one in degrees Rankine (the Fahrenheit analogue of Kelvins). This is one
of the few examples of recognized scientific units that have both a
scale and an offset relative to each other.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Ticker
from matplotlib.ticker import (
    TransformFormatter, StrMethodFormatter, MaxNLocator
)


class LinearTransform:
    """
    A callable class that transforms input values to output according to
    a linear transformation.
    """

    def __init__(self, in_start=0.0, in_end=None, out_start=0.0, out_end=None):
        """
        Sets up the transformation such that `in_start` gets mapped to
        `out_start` and `in_end` gets mapped to `out_end`. The following
        shortcuts apply when only some of the inputs are specified:

          - none: no-op
          - in_start: translation to zero
          - out_start: translation from zero
          - in_end: scaling to one (divide input by in_end)
          - out_end: scaling from one (multiply input by in_end)
          - in_start, out_start: translation
          - in_end, out_end: scaling (in_start and out_start zero)
          - in_start, out_end: in_end=out_end, out_start=0
          - in_end, out_start: in_start=0, out_end=in_end

        Based on the following rules:

          - start missing: set start to zero
          - both ends are missing: set ranges to 1.0
          - one end is missing: set it to the other end
        """
        if in_end is not None:
            in_scale = in_end - in_start
        elif out_end is not None:
            in_scale = out_end - in_start
        else:
            in_scale = 1.0

        if out_end is not None:
            out_scale = out_end - out_start
        elif in_end is not None:
            out_scale = in_end - out_start
        else:
            out_scale = 1.0

        self._scale = out_scale / in_scale
        self._offset = out_start - self._scale * in_start

    def __call__(self, x):
        """
        Transforms the input value `x` according to the rule set up in
        `__init__`.
        """
        return x * self._scale + self._offset

# X-data
temp_C = np.arange(-5.0, 5.1, 0.25)
# Y-data
counts = 15.0 * np.exp(-temp_C**2 / 25)
# Add some noise
counts += np.random.normal(scale=4.0, size=counts.shape)
if counts.min() < 0:
    counts += counts.min()

fig, ax1 = plt.subplots()
ax2 = fig.add_subplot(111, sharex=ax1, sharey=ax1, frameon=False)

ax1.plot(temp_C, counts, drawstyle='steps-mid')

ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:0.2f}'))

# This step is necessary to allow the shared x-axes to have different
# Formatter and Locator objects.
ax2.xaxis.major = Ticker()
# 0C -> 491.67R (definition), -273.15C (0K)->0R (-491.67F)(definition)
ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
ax2.xaxis.set_major_formatter(
        TransformFormatter(LinearTransform(in_start=-273.15, in_end=0,
                                           out_end=491.67),
                           StrMethodFormatter('{x:0.2f}')))

# The y-axes share their locators and formatters, so only one needs to
# be set
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
# Setting the transfrom to `int` will only alter the type, not the
# actual value of the ticks
ax1.yaxis.set_major_formatter(
        TransformFormatter(int, StrMethodFormatter('{x:02X}')))

ax1.set_xlabel('Temperature (\u00B0C)')
ax1.set_ylabel('Samples (Hex)')
ax2.set_xlabel('Temperature (\u00B0R)')

ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')

plt.show()
