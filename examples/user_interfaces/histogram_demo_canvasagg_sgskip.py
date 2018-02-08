"""
========================
Histogram Demo CanvasAgg
========================

This is an example that shows you how to work directly with the agg
figure canvas to create a figure using the pythonic API.

In this example, the contents of the agg canvas are extracted to a
string, which can in turn be passed off to PIL or put in a numeric
array


"""
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)

canvas = FigureCanvasAgg(fig)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = ax.hist(x, 50, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
line, = ax.plot(bins, y, 'r--')
line.set_linewidth(1)

ax.set_xlabel('Smarts')
ax.set_ylabel('Probability')
ax.set_title(r'$\mathrm{Histogram of IQ: }\mu=100, \sigma=15$')

ax.set_xlim((40, 160))
ax.set_ylim((0, 0.03))

canvas.draw()

s = canvas.tostring_rgb()  # save this and convert to bitmap as needed

# Get the figure dimensions for creating bitmaps or NumPy arrays,
# etc.
l, b, w, h = fig.bbox.bounds
w, h = int(w), int(h)

if 0:
    # Convert to a NumPy array
    X = np.fromstring(s, np.uint8).reshape((h, w, 3))

if 0:
    # pass off to PIL
    from PIL import Image
    im = Image.fromstring("RGB", (w, h), s)
    im.show()
