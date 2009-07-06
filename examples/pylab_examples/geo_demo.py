import numpy as np
#np.seterr("raise")

from pylab import *

subplot(221, projection="aitoff")
title("Aitoff")
grid(True)

subplot(222, projection="hammer")
title("Hammer")
grid(True)

subplot(223, projection="lambert")
title("Lambert")
grid(True)

subplot(224, projection="mollweide")
title("Mollweide")
grid(True)

show()
