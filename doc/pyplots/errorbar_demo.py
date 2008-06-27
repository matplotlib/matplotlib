import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.1, 4, 0.1)
s = np.exp(-t)
e, f = 0.1*np.absolute(np.random.randn(2, len(s)))
plt.errorbar(t, s, e, fmt='o')             # vertical symmetric
plt.show()
