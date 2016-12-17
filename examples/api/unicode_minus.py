"""
=============
Unicode minus
=============

You can use the proper typesetting Unicode minus (see
https://en.wikipedia.org/wiki/Plus_sign#Plus_sign) or the ASCII hyphen
for minus, which some people prefer.  The matplotlibrc param
axes.unicode_minus controls the default behavior.

The default is to use the Unicode minus.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
ax.plot(10*np.random.randn(100), 10*np.random.randn(100), 'o')
ax.set_title('Using hyphen instead of Unicode minus')
plt.show()
