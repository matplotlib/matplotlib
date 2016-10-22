"""
================================================================
Creating sample data for the different examples on normalization
================================================================

Data with special features tailored to the need of the different examples on
colormal normalization is created.

"""

import numpy as np


def PiecewiseNormData(NX=512, NY=256):
    """Sample data for the PiecewiseNorm class.

    Returns a 2d array with sample data, along with the X and Y values for the
    array.

    Parameters
    ----------
    NX : int
        Number of samples for the data accross the horizontal dimension.
        Default is 512.
    NY : int
        Number of samples for the data accross the vertical dimension.
        Default is 256.

    Returns
    -------
    X, Y, data : ndarray of shape (NX,NY)
        Values for the `X` coordinates, the `Y` coordinates, and the `data`.

    Examples
    --------
    >>> X,Y,Z=PiecewiseNormData()
    """

    xmax = 16 * np.pi
    x = np.linspace(0, xmax, NX)
    y = np.linspace(-2, 2, NY)
    X, Y = np.meshgrid(x, y)

    data = np.zeros(X.shape)

    def gauss2d(x, y, a0, x0, y0, wx, wy):
        return a0 * np.exp(-(x - x0)**2 / wx**2 - (y - y0)**2 / wy**2)

    maskY = (Y > -1) * (Y <= 0)
    N = 31
    for i in range(N):
        maskX = (X > (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
        mask = maskX * maskY
        data[mask] += gauss2d(X[mask], Y[mask], 2. * i / (N - 1), (i + 0.5) *
                              (xmax / N), -0.25, xmax / (3 * N), 0.07)
        data[mask] -= gauss2d(X[mask], Y[mask], 1. * i / (N - 1), (i + 0.5) *
                              (xmax / N), -0.75, xmax / (3 * N), 0.07)

    maskY = (Y > 0) * (Y <= 1)
    data[maskY] = np.cos(X[maskY]) * Y[maskY]**2

    N = 61
    maskY = (Y > 1) * (Y <= 2.)
    for i, val in enumerate(np.linspace(-1, 1, N)):
        if val < 0:
            aux = val
        if val > 0:
            aux = val * 2

        maskX = (X >= (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
        data[maskX * maskY] = aux

    N = 11
    maskY = (Y <= -1)
    for i, val in enumerate(np.linspace(-1, 1, N)):
        if val < 0:
            factor = 1
        if val >= 0:
            factor = 2
        maskX = (X >= (i * (xmax / N))) * (X <= ((i + 1) * (xmax / N)))
        mask = maskX * maskY
        data[mask] = val * factor

        if i != N - 1:
            data[mask] += gauss2d(X[mask], Y[mask], 0.05 * factor, (i + 0.5) *
                                  (xmax / N), -1.25, xmax / (3 * N), 0.07)
        if i != 0:
            data[mask] -= gauss2d(X[mask], Y[mask], 0.05 * factor, (i + 0.5) *
                                  (xmax / N), -1.75, xmax / (3 * N), 0.07)
    return X, Y, data
