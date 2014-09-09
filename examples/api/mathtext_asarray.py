"""
Load a mathtext image as numpy array
"""

import matplotlib.mathtext as mathtext
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('image', origin='upper')

parser = mathtext.MathTextParser("Bitmap")
parser.to_png('test2.png',
              r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} '
              r'y\right)\right]$', color='green', fontsize=14, dpi=100)

rgba1, depth1 = parser.to_rgba(
    r'IQ: $\sigma_i=15$', color='blue', fontsize=20, dpi=200)
rgba2, depth2 = parser.to_rgba(
    r'some other string', color='red', fontsize=20, dpi=200)

fig = plt.figure()
fig.figimage(rgba1.astype(float)/255., 100, 100)
fig.figimage(rgba2.astype(float)/255., 100, 300)

plt.show()
