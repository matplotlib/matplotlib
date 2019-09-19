import matplotlib.pyplot as plt
import numpy as np
f = 3
t = np.linspace(0, 1, 100)
s = np.sin(2 * np.pi * f * t)
plt.plot(t, s)
