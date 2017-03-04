import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


x = randn(5000)

# Create a figure with some axes. This makes it easier to set the
# formatter later, since that is only available through the OO API.
fig, ax = plt.subplots()

# Make a normed histogram. It'll be multiplied by 100 later.
ax.hist(x, bins=50, normed=True)

# Set the formatter. `xmax` sets the value that maps to 100%.
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

plt.show()
