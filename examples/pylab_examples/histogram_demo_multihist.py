import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flat

mu, sigma = 200, 25
x = mu + sigma*np.random.randn(1000,3)

n, bins, patches = ax0.hist(x, 10, normed=1, histtype='bar',
                            color=['crimson', 'burlywood', 'chartreuse'],
                            label=['Crimson', 'Burlywood', 'Chartreuse'])
ax0.legend(prop={'size': 10})


n, bins, patches = ax1.hist(x, 10, normed=1, histtype='bar', stacked=True)

n, bins, patches = ax2.hist(x, 10, histtype='step', stacked=True, fill=True)

# Make a multiple-histogram of data-sets with different length.
x0 = mu + sigma*np.random.randn(10000)
x1 = mu + sigma*np.random.randn(7000)
x2 = mu + sigma*np.random.randn(3000)

w0 = np.ones_like(x0)
w0[:len(x0)/2] = 0.5
w1 = np.ones_like(x1)
w1[:len(x1)/2] = 0.5
w2 = np.ones_like(x2)
w2[:len(x2)/2] = 0.5

n, bins, patches = ax3.hist( [x0,x1,x2], 10, weights=[w0, w1, w2], histtype='bar')

plt.tight_layout()
plt.show()
