import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flat

mu, sigma = 200, 25
x = mu + sigma*np.random.randn(1000,3)

colors = ['crimson', 'burlywood', 'chartreuse']
n, bins, patches = ax0.hist(x, 10, normed=1, histtype='bar',
                            color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bar')


n, bins, patches = ax1.hist(x, 10, normed=1, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

n, bins, patches = ax2.hist(x, 10, histtype='step', stacked=True, fill=True)
ax2.set_title('stepfilled')

# Make a multiple-histogram of data-sets with different length.
x_multi = [mu + sigma*np.random.randn(n) for n in [10000, 5000, 2000]]

n, bins, patches = ax3.hist(x_multi, 10, histtype='bar')
ax3.set_title('different sample sizes')

plt.tight_layout()
plt.show()
