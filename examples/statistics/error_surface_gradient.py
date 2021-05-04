"""
======================================================
Plot an error surface of gradient descent
======================================================
This example shows how to plot an error surface for the 
descending gradient of a two-dimensional dataset, using 
the quadratic error as an error function. The gradient 
method is used to minimize a given function to its local 
minimum.
"""

import random, math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

def error(X, y, w):
    return 0.5 * np.linalg.norm(X.dot(w) - y)**2

def linear_regression_gradient(X, y, w):
    return X.T.dot(X.dot(w)-y)

def plot_mesh(X_data, y_data, bounds):
    (minx,miny),(maxx,maxy) = bounds
    
    x_range = np.linspace(minx, maxx, num=50)
    y_range = np.linspace(miny, maxy, num=50)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros((len(x_range), len(y_range)))
    
    for i, w_i in enumerate(x_range):
        for j, w_j in enumerate(y_range):
            Z[j,i] = error(X_data, y_data, [w_i, w_j])
    fig = plt.figure(figsize=(7,5))
    ax = fig.gca(projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=1, antialiased=True, alpha=0.5)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, cmap=cm.plasma, linewidth=1, color='navy')
    return Z

    
X = np.array([[0.], [2.], [4.], [6.], [8.], [10.]])
y = np.array([22.5, 6.0, 4.0, 3.5, 2.2, 1.])  

X_bias = np.hstack((X, np.ones((len(X), 1))))

w = np.random.random(2)/10000

alpha = 0.05
epoch = 50

for i in range(epoch):
    linear_regression = linear_regression_gradient(X_bias, y, w)
    w = w - alpha * linear_regression
    
plot_mesh(X_bias, y, [[-7.5, -10], [5, 15]])
plt.setp(plt.gca(), xlabel='$Weight_1$', ylabel='$Weight_2$', zlabel='$\mathrm{E}(Weight_1,Weight_2)$')
plt.tight_layout()
