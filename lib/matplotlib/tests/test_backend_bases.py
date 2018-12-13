import re

from matplotlib.backend_bases import (
    FigureCanvasBase, LocationEvent, RendererBase)
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.path as path
import os
import numpy as np
import pytest


def test_uses_per_path():
    id = transforms.Affine2D()
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    tforms = [id.rotate(i) for i in range(1, 5)]
    offsets = np.arange(20).reshape((10, 2))
    facecolors = ['red', 'green']
    edgecolors = ['red', 'green']

    def check(master_transform, paths, all_transforms,
              offsets, facecolors, edgecolors):
        rb = RendererBase()
        raw_paths = list(rb._iter_collection_raw_paths(
            master_transform, paths, all_transforms))
        gc = rb.new_gc()
        ids = [path_id for xo, yo, path_id, gc0, rgbFace in
               rb._iter_collection(gc, master_transform, all_transforms,
                                   range(len(raw_paths)), offsets,
                                   transforms.IdentityTransform(),
                                   facecolors, edgecolors, [], [], [False],
                                   [], 'data')]
        uses = rb._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        if raw_paths:
            seen = np.bincount(ids, minlength=len(raw_paths))
            assert set(seen).issubset([uses - 1, uses])

    check(id, paths, tforms, offsets, facecolors, edgecolors)
    check(id, paths[0:1], tforms, offsets, facecolors, edgecolors)
    check(id, [], tforms, offsets, facecolors, edgecolors)
    check(id, paths, tforms[0:1], offsets, facecolors, edgecolors)
    check(id, paths, [], offsets, facecolors, edgecolors)
    for n in range(0, offsets.shape[0]):
        check(id, paths, tforms, offsets[0:n, :], facecolors, edgecolors)
    check(id, paths, tforms, offsets, [], edgecolors)
    check(id, paths, tforms, offsets, facecolors, [])
    check(id, paths, tforms, offsets, [], [])
    check(id, paths, tforms, offsets, facecolors[0:1], edgecolors)


def test_get_default_filename(tmpdir):
    plt.rcParams['savefig.directory'] = str(tmpdir)
    fig = plt.figure()
    canvas = FigureCanvasBase(fig)
    filename = canvas.get_default_filename()
    assert filename == 'image.png'


@pytest.mark.backend('pdf')
def test_non_gui_warning(monkeypatch):
    plt.subplots()

    monkeypatch.setitem(os.environ, "DISPLAY", ":999")

    with pytest.warns(UserWarning) as rec:
        plt.show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))

    with pytest.warns(UserWarning) as rec:
        plt.gcf().show()
        assert len(rec) == 1
        assert ('Matplotlib is currently using pdf, which is a non-GUI backend'
                in str(rec[0].message))


@pytest.mark.parametrize(
    "x, y", [(42, 24), (None, 42), (None, None), (200, 100.01), (205.75, 2.0)])
def test_location_event_position(x, y):
    # LocationEvent should cast its x and y arguments to int unless it is None.
    fig, ax = plt.subplots()
    canvas = FigureCanvasBase(fig)
    event = LocationEvent("test_event", canvas, x, y)
    if x is None:
        assert event.x is None
    else:
        assert event.x == int(x)
        assert isinstance(event.x, int)
    if y is None:
        assert event.y is None
    else:
        assert event.y == int(y)
        assert isinstance(event.y, int)
    if x is not None and y is not None:
        assert re.match(
            "x={} +y={}".format(ax.format_xdata(x), ax.format_ydata(y)),
            ax.format_coord(x, y))
        ax.fmt_xdata = ax.fmt_ydata = lambda x: "foo"
        assert re.match("x=foo +y=foo", ax.format_coord(x, y))
