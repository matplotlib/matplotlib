import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 200, 25
plt.figure()

x = mu + sigma*np.random.randn(1000,3)

n, bins, patches = plt.hist(x, 10, normed=1, histtype='bar',
                            color=['crimson', 'burlywood', 'chartreuse'],
                            label=['Crimson', 'Burlywood', 'Chartreuse'])
plt.legend()


plt.figure()
n, bins, patches = plt.hist(x, 10, normed=1, histtype='bar', stacked=True)

plt.figure()
n, bins, patches = plt.hist(x, 10, histtype='step', stacked=True, fill=True)

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


plt.figure()

n, bins, patches = plt.hist( [x0,x1,x2], 10, weights=[w0, w1, w2], histtype='bar')

plt.show()
