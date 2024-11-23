import numpy as np

import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def test_scatter_3d_projection_conservation():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fix axes3d projection
    ax.roll = 0
    ax.elev = 0
    ax.azim = -45
    ax.stale = True

    x = [0, 1, 2, 3, 4]
    scatter_collection = ax.scatter(x, x, x)
    fig.canvas.draw_idle()

    # Get scatter location on canvas and freeze the data
    scatter_offset = scatter_collection.get_offsets()
    scatter_location = ax.transData.transform(scatter_offset)

    # Yaw -44 and -46 are enough to produce two set of scatter
    # with opposite z-order without moving points too far
    for azim in (-44, -46):
        ax.azim = azim
        ax.stale = True
        fig.canvas.draw_idle()

        for i in range(5):
            # Create a mouse event used to locate and to get index
            # from each dots
            event = MouseEvent("button_press_event", fig.canvas,
                               *scatter_location[i, :])
            contains, ind = scatter_collection.contains(event)
            assert contains is True
            assert len(ind["ind"]) == 1
            assert ind["ind"][0] == i


def test_zordered_error():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/26497
    lc = [(np.fromiter([0.0, 0.0, 0.0], dtype="float"),
           np.fromiter([1.0, 1.0, 1.0], dtype="float"))]
    pc = [np.fromiter([0.0, 0.0], dtype="float"),
          np.fromiter([0.0, 1.0], dtype="float"),
          np.fromiter([1.0, 1.0], dtype="float")]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.add_collection(Line3DCollection(lc))
    ax.scatter(*pc, visible=False)
    plt.draw()


def test_generate_normals():

    # Following code is an example taken from
    # https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib
    # and modified to test _generate_normals function

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [0, 2, 1, 1]
    y = [0, 0, 1, 0]
    z = [0, 0, 0, 1]

    # deliberately use nested tuple
    vertices = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    tupleList = list(zip(x, y, z))

    poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))]
              for ix in range(len(vertices))]
    ax.scatter(x, y, z)
    collection = Poly3DCollection(poly3d, alpha=0.2, edgecolors='r', shade=True)
    face_color = [0.5, 0.5, 1]  # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)

    plt.draw()
