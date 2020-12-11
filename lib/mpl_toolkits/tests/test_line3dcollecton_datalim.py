from mpl_toolkits.mplot3d import art3d
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


def test_get_datalim1():
    segments = [[(0, 0, 0), (1, 1, 1)]]
    collection = art3d.Line3DCollection(segments)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection)
    result = collection.get_datalim(ax.transData).get_points()
    correct = mpl.transforms.Bbox([[0, 0], [1, 1]]).get_points()
    np.testing.assert_almost_equal(result, correct)


def test_get_datalim2():
    segments = [[(0, 0, 0), (1, 1, 1), (0, 0, 0)], [(0, 0, 0), (1, 2, 2)]]
    collection = art3d.Line3DCollection(segments)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection)
    result = collection.get_datalim(ax.transData).get_points()
    correct = mpl.transforms.Bbox([[0, 0], [1, 2]]).get_points()
    np.testing.assert_almost_equal(result, correct)


def test_get_datalim3():
    z = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    x = np.sin(z)
    y = np.cos(z)
    segments3D = []
    segments2D = []
    for i in range(len(z)-1):
        z0, x0, y0 = z[i], x[i], y[i]
        z1, x1, y1 = z[i+1], x[i+1], y[i+1]
        segments3D.append([(x0, y0, z0), (x1, y1, z1)])
        segments2D.append([(x0, y0), (x1, y1)])

    collection3D = art3d.Line3DCollection(segments3D)
    collection2D = LineCollection(segments2D)
    fig, ax = plt.subplots()
    ax.add_collection(collection2D)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    correct = collection2D.get_datalim(ax.transData).get_points()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection3D)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    result = collection3D.get_datalim(ax.transData).get_points()
    np.testing.assert_almost_equal(result, correct)


def test_get_datalim_setLims():
    z = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    x = np.sin(z)
    y = np.cos(z)
    segments3D = []
    segments2D = []
    for i in range(len(z)-1):
        z0, x0, y0 = z[i], x[i], y[i]
        z1, x1, y1 = z[i+1], x[i+1], y[i+1]
        segments3D.append([(x0, y0, z0), (x1, y1, z1)])
        segments2D.append([(x0, y0), (x1, y1)])

    collection3D = art3d.Line3DCollection(segments3D)
    collection2D = LineCollection(segments2D)
    fig, ax = plt.subplots()
    ax.add_collection(collection2D)
    correct = collection2D.get_datalim(ax.transData).get_points()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection3D)
    result = collection3D.get_datalim(ax.transData).get_points()
    np.testing.assert_almost_equal(result, correct)


def test_get_datalim4():
    segments3D = [[(0, 0, 0), (10, 10, 1)]]
    segments2D = [[(0, 0), (10, 10)]]

    collection3D = art3d.Line3DCollection(segments3D)
    collection2D = LineCollection(segments2D)
    fig, ax = plt.subplots()
    ax.add_collection(collection2D)
    correct = collection2D.get_datalim(ax.transData).get_points()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection3D)
    result = collection3D.get_datalim(ax.transData).get_points()
    ax.set_xlim(0, result[1][0])
    ax.set_ylim(0, result[1][1])
    np.testing.assert_almost_equal(result, correct)


def test_get_paths():
    segments3D = [[(0, 0, 0), (1, 1, 1)]]
    collection3D = art3d.Line3DCollection(segments3D)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection3D)
    result = collection3D.get_paths()
    assert len(result) > 0


def test_get_datalim_random():
    np.random.seed(19680801)
    num_segments = np.random.randint(0, 10000)
    segments3D = [[]]
    segments2D = [[]]
    for i in range(0, num_segments):
        #append random x, y and z values
        x = np.random.randint(0, 10000)
        y = np.random.randint(0, 10000)
        z = np.random.randint(0, 10000)
        segments3D[0].append((x, y, x))
        segments2D[0].append((x, y))

    collection3D = art3d.Line3DCollection(segments3D)
    collection2D = LineCollection(segments2D)
    fig, ax = plt.subplots()
    ax.add_collection(collection2D)
    correct = collection2D.get_datalim(ax.transData).get_points()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.add_collection3d(collection3D)
    result = collection3D.get_datalim(ax.transData).get_points()
    np.testing.assert_almost_equal(result, correct)
