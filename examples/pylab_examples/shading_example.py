import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import lightsource

# example showing how to make shaded relief plots 
# like mathematica
# (http://reference.wolfram.com/mathematica/ref/ReliefPlot.html )
# or Generic Mapping Tools
# (http://gmt.soest.hawaii.edu/gmt/doc/gmt/html/GMT_Docs/node145.html)

# test data
X,Y=np.mgrid[-5:5:0.1,-5:5:0.1]
Z=X+np.sin(X**2+Y**2)
# creat light source object.
ls = lightsource(azdeg=270,altdeg=60)
# shade data, creating an rgb array.
rgb = ls.shade(Z,plt.cm.copper)
# plot un-shaded and shaded images.
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.imshow(Z,cmap=plt.cm.copper)
plt.title('imshow')
plt.xticks([]); plt.yticks([])
plt.subplot(122)
plt.imshow(rgb)
plt.title('imshow with shading')
plt.xticks([]); plt.yticks([])
plt.show()
