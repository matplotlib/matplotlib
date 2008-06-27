import numpy as np
import matplotlib.pyplot as plt

plt.subplots_adjust(hspace=0.4)
t = np.arange(0.01, 20.0, 0.01)

# log y axis
plt.subplot(311)
plt.semilogy(t, np.exp(-t/5.0))
plt.ylabel('semilogy')
plt.grid(True)

# log x axis
plt.subplot(312)
plt.semilogx(t, np.sin(2*np.pi*t))
plt.ylabel('semilogx')
plt.grid(True)

# log x and y axis
plt.subplot(313)
plt.loglog(t, 20*np.exp(-t/10.0), basex=4)
plt.grid(True)
plt.ylabel('loglog base 4 on x')

plt.show()
