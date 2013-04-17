import matplotlib.pyplot as plt
from numpy.random import rand

Z = rand(6,10)

plt.subplot(2,1,1)
c = plt.pcolor(Z)
plt.title('default: no edges')

plt.subplot(2,1,2)
c = plt.pcolor(Z, edgecolors='k', linewidths=4)
plt.title('thick edges')

plt.show()
