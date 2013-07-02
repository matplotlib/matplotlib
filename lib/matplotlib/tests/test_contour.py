import numpy as np

from matplotlib.testing.decorators import cleanup, image_comparison
from matplotlib import pyplot as plt

import re


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
        assert re.match(
            r'Shape of x does not match that of z: ' +
            r'found \(9L?, 9L?\) instead of \(9L?, 10L?\)\.',
            exc.args[0]) is not None

    try:
        ax.contour(g, b, z)
    except TypeError as exc:
        assert re.match(
            r'Shape of y does not match that of z: ' +
            r'found \(9L?, 9L?\) instead of \(9L?, 10L?\)\.',
            exc.args[0]) is not None


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


@image_comparison(baseline_images=['contour_manual_labels'])
def test_contour_manual_labels():

    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)

    plt.figure(figsize=(6, 2))
    cs = plt.contour(x,y,z)
    pts = np.array([(1.5, 3.0), (1.5, 4.4), (1.5, 6.0)])
    plt.clabel(cs, manual=pts)


@image_comparison(baseline_images=['contour_manual_colors_and_levels'],
                  extensions=['png'], remove_text=True)
def test_given_colors_levels_and_extends():
    _, axes = plt.subplots(2, 4)

    data = np.arange(12).reshape(3, 4)

    colors = ['red', 'yellow', 'pink', 'blue', 'black']
    levels = [2, 4, 8, 10]

    for i, ax in enumerate(axes.flatten()):
        plt.sca(ax)

        filled = i % 2 == 0.
        extend = ['neither', 'min', 'max', 'both'][i // 2]

        if filled:
            last_color = -1 if extend in ['min', 'max'] else None
            plt.contourf(data, colors=colors[:last_color], levels=levels, extend=extend)
        else:
            last_level = -1 if extend == 'both' else None
            plt.contour(data, colors=colors, levels=levels[:last_level], extend=extend)

        plt.colorbar()


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
