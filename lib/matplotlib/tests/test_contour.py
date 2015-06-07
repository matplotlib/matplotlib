from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import datetime

import numpy as np
from matplotlib import mlab
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
        print(exc.args[0])
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
    cs = plt.contour(x, y, z)
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
            plt.contourf(data, colors=colors[:last_color], levels=levels,
                         extend=extend)
        else:
            last_level = -1 if extend == 'both' else None
            plt.contour(data, colors=colors, levels=levels[:last_level],
                        extend=extend)

        plt.colorbar()


@image_comparison(baseline_images=['contour_datetime_axis'],
                  extensions=['png'], remove_text=False)
def test_contour_datetime_axis():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(20)])
    y = np.arange(20)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.contour(x, y, z)
    plt.subplot(222)
    plt.contourf(x, y, z)
    x = np.repeat(x[np.newaxis], 20, axis=0)
    y = np.repeat(y[:, np.newaxis], 20, axis=1)
    plt.subplot(223)
    plt.contour(x, y, z)
    plt.subplot(224)
    plt.contourf(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


@image_comparison(baseline_images=['contour_test_label_transforms'],
                  extensions=['png'], remove_text=True)
def test_labels():
    # Adapted from pylab_examples example code: contour_demo.py
    # see issues #2475, #2843, and #2818 for explanation
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)

    fig, ax = plt.subplots(1, 1)
    CS = ax.contour(X, Y, Z)
    disp_units = [(216, 177), (359, 290), (521, 406)]
    data_units = [(-2, .5), (0, -1.5), (2.8, 1)]

    CS.clabel()

    for x, y in data_units:
        CS.add_label_near(x, y, inline=True, transform=None)

    for x, y in disp_units:
        CS.add_label_near(x, y, inline=True, transform=False)


@image_comparison(baseline_images=['contour_corner_mask_False',
                                   'contour_corner_mask_True'],
                  extensions=['png'], remove_text=True)
def test_corner_mask():
    n = 60
    mask_level = 0.95
    noise_amp = 1.0
    np.random.seed([1])
    x, y = np.meshgrid(np.linspace(0, 2.0, n), np.linspace(0, 2.0, n))
    z = np.cos(7*x)*np.sin(8*y) + noise_amp*np.random.rand(n, n)
    mask = np.where(np.random.rand(n, n) >= mask_level, True, False)
    z = np.ma.array(z, mask=mask)

    for corner_mask in [False, True]:
        fig = plt.figure()
        plt.contourf(z, corner_mask=corner_mask)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
