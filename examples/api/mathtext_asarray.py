"""
Load a mathtext image as numpy array
"""

import numpy as np
import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt

parser = mathtext.MathTextParser("Bitmap")

parser.to_png('test2.png', r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$', color='green', fontsize=14, dpi=100)


rgba = parser.to_rgba(r'IQ: $\sigma_i=15$', color='blue', fontsize=20, dpi=200)

fig = plt.figure()
fig.figimage(rgba.astype(float)/255., 100, 100)

plt.show()
