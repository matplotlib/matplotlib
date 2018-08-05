"""
==============
Secondary Axis
==============

Sometimes we want as secondary axis on a plot, for instance to convert
radians to degrees on the same plot.  We can do this by making a child
axes with only one axis visible via `.Axes.axes.secondary_xaxis` and
`.Axes.axes.secondary_yaxis`.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Transform

fig, ax = plt.subplots(constrained_layout=True)
x = np.arange(0, 360, 1)
y = np.sin(2 * x * np.pi / 180)
ax.plot(x, y)
ax.set_xlabel('angle [degrees]')
ax.set_ylabel('signal')
ax.set_title('Sine wave')

secax = ax.secondary_xaxis('top', conversion=[np.pi / 180])
secax.set_xlabel('angle [rad]')
plt.show()

###########################################################################
# The conversion can be a linear slope and an offset as a 2-tuple.  It can
# also be more complicated.  The strings  "inverted", "power", and "linear"
# are accepted as valid arguments for the ``conversion`` kwarg, and scaling
# is set by the ``otherargs`` kwarg.
#
# .. note ::
#
#   In this case, the xscale of the parent is logarithmic, so the child is
#   made logarithmic as well.

fig, ax = plt.subplots(constrained_layout=True)
x = np.arange(0.02, 1, 0.02)
np.random.seed(19680801)
y = np.random.randn(len(x)) ** 2
ax.loglog(x, y)
ax.set_xlabel('f [Hz]')
ax.set_ylabel('PSD')
ax.set_title('Random spetrum')

secax = ax.secondary_xaxis('top', conversion='inverted', otherargs=1)
secax.set_xlabel('period [s]')
secax.set_xscale('log')
plt.show()

###########################################################################
# Considerably more complicated, the user can define their own transform
# to pass to ``conversion``:

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.arange(2, 11), np.arange(2, 11))


class LocalInverted(Transform):
    """
    Return a/x
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, fac):
        Transform.__init__(self)
        self._fac = fac

    def transform_non_affine(self, values):
        with np.errstate(divide="ignore", invalid="ignore"):
            q = self._fac / values
        return q

    def inverted(self):
        """ we are just our own inverse """
        return LocalInverted(1 / self._fac)

secax = ax.secondary_xaxis('top', conversion="inverted", otherargs=1)

plt.show()
