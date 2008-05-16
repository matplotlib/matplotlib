import numpy as np
np.seterr("raise")

from pylab import *

subplot(221, projection="aitoff")
grid(True)

subplot(222, projection="hammer")
grid(True)

subplot(223, projection="lambert")
grid(True)


show()
