import numpy as np

from matplotlib.testing.decorators import cleanup
from matplotlib import pyplot as plt


@cleanup
def test_contour_shape_1d_valid():

    x = np.arange(10)
    y = np.arange(9)
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(x, y, z)


@cleanup
def test_contour_shape_2d_valid():

    x = np.arange(10)
    y = np.arange(9)
    xg, yg = np.meshgrid(x, y)
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(xg, yg, z)


@cleanup
def test_contour_shape_mismatch_1():

    x = np.arange(9)
    y = np.arange(9)
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(x, y, z)
    except TypeError as exc:
        assert exc.args[0] == 'Length of x must be number of columns in z.'


@cleanup
def test_contour_shape_mismatch_2():

    x = np.arange(10)
    y = np.arange(10)
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(x, y, z)
    except TypeError as exc:
        assert exc.args[0] == 'Length of y must be number of rows in z.'


@cleanup
def test_contour_shape_mismatch_3():

    x = np.arange(10)
    y = np.arange(10)
    xg, yg = np.meshgrid(x, y)
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(xg, y, z)
    except TypeError as exc:
        assert exc.args[0] == 'Number of dimensions of x and y should match.'

    try:
        ax.contour(x, yg, z)
    except TypeError as exc:
        assert exc.args[0] == 'Number of dimensions of x and y should match.'


@cleanup
def test_contour_shape_mismatch_4():

    g = np.random.random((9, 10))
    b = np.random.random((9, 9))
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(b, g, z)
    except TypeError as exc:
        print exc.args[0]
        assert exc.args[0] == 'Shape of x does not match that of z: ' + \
                              'found (9, 9) instead of (9, 10).'

    try:
        ax.contour(g, b, z)
    except TypeError as exc:
        assert exc.args[0] == 'Shape of y does not match that of z: ' + \
                              'found (9, 9) instead of (9, 10).'


@cleanup
def test_contour_shape_invalid_1():

    x = np.random.random((3, 3, 3))
    y = np.random.random((3, 3, 3))
    z = np.random.random((9, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(x, y, z)
    except TypeError as exc:
        assert exc.args[0] == 'Inputs x and y must be 1D or 2D.'


@cleanup
def test_contour_shape_invalid_2():

    x = np.random.random((3, 3, 3))
    y = np.random.random((3, 3, 3))
    z = np.random.random((3, 3, 3))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    try:
        ax.contour(x, y, z)
    except TypeError as exc:
        assert exc.args[0] == 'Input z must be a 2D array.'
