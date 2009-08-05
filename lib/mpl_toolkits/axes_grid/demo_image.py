import numpy as np

def get_demo_image():
    # prepare image
    delta = 0.5

    extent = (-3,4,-4,3)
    x = np.arange(-3.0, 4.001, delta)
    y = np.arange(-4.0, 3.001, delta)
    X, Y = np.meshgrid(x, y)
    import matplotlib.mlab as mlab
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = (Z1 - Z2) * 10

    return Z, extent

