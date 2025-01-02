"""
===================================================
SI prefixed offsets and natural order of magnitudes
===================================================

`matplotlib.ticker.EngFormatter` is capable of computing a natural
offset for your axis data, and presenting it with a standard SI prefix
automatically calculated.

Below is an examples of such a plot:

"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as mticker

# Fixing random state for reproducibility
np.random.seed(19680801)

UNIT = "Hz"

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(mticker.EngFormatter(
    useOffset=True,
    unit=UNIT
))
size = 100
measurement = np.full(size, 1e9)
noise = np.random.uniform(low=-2e3, high=2e3, size=size)
ax.plot(measurement + noise)
plt.show()
