"""
This example demonstrates the "bmh" style, which is the design used in the 
Bayesian Methods for Hackers online book.

"""
from numpy.random import beta
import matplotlib.pyplot as plt

plt.style.use('bmh')
plt.hist(beta(10,10,size=10000),histtype="stepfilled", bins=25, alpha=0.8, normed=True)
plt.hist(beta(4,12,size=10000),histtype="stepfilled", bins=25, alpha=0.7, normed=True)
plt.hist(beta(50,12,size=10000),histtype="stepfilled", bins=25, alpha=0.8, normed=True)
plt.hist(beta(6,55,size=10000),histtype="stepfilled", bins=25, alpha=0.8, normed=True)

plt.show()