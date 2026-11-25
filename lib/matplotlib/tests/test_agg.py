import io
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import features, Image, TiffTags
import pytest


from matplotlib import (
    collections, colors as mcolors, patheffects, pyplot as plt,
    transforms as mtransforms,
    rcParams, rc_context)
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import IdentityTransform


def require_pillow_feature(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        available = features.check(name.lower())
    return pytest.mark.skipif(not available, reason=f"{name} support not available")


def test_repeated_save_with_alpha():
    # We want an image which has a background color of bluish green, with an
    # alpha of 0.25.

    fig = Figure([1, 0.4])
    fig.set_facecolor((0, 1, 0.4))
    fig.patch.set_alpha(0.25)

    # The target color is fig.patch.get_facecolor()

    buf = io.BytesIO()

    fig.savefig(buf,
                facecolor=fig.get_facecolor(),
                edgecolor='none')

    # Save the figure again to check that the
    # colors don't bleed from the previous renderer.
    buf.seek(0)
    fig.savefig(buf,
                facecolor=fig.get_facecolor(),
                edgecolor='none')

    # Check the first pixel has the desired color & alpha
    # (approx: 0, 1.0, 0.4, 0.25)
    buf.seek(0)
    assert_array_almost_equal(tuple(imread(buf)[0, 0]),
                              (0.0, 1.0, 0.4, 0.250),
                              decimal=3)


def test_large_single_path_collection():
    buff = io.BytesIO()

    # Generates a too-large single path in a path collection that
    # would cause a segfault if the draw_markers optimization is
    # applied.
    f, ax = plt.subplots()
    collection = collections.PathCollection(
        [Path([[-10, 5], [10, 5], [10, -5], [-10, -5], [-10, 5]])])
    ax.add_artist(collection)
    ax.set_xlim(10**-3, 1)
    plt.savefig(buff)


def test_marker_with_nan():
    # This creates a marker with nans in it, which was segfaulting the
    # Agg backend (see #3722)
    fig, ax = plt.subplots(1)
    steps = 1000
    data = np.arange(steps)
    ax.semilogx(data)
    ax.fill_between(data, data*0.8, data*1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')


def test_long_path():
    buff = io.BytesIO()
    fig = Figure()
    ax = fig.subplots()
    points = np.ones(100_000)
    points[::2] *= -1
    ax.plot(points)
    fig.savefig(buff, format='png')


@image_comparison(['agg_filter.png'], remove_text=True, style='_classic_test')
def test_agg_filter():
    def smooth1d(x, window_len):
        # copied from https://scipy-cookbook.readthedocs.io/
        s = np.r_[
            2*x[0] - x[window_len:1:-1], x, 2*x[-1] - x[-1:-window_len:-1]]
        w = np.hanning(window_len)
        y = np.convolve(w/w.sum(), s, mode='same')
        return y[window_len-1:-window_len+1]

    def smooth2d(A, sigma=3):
        window_len = max(int(sigma), 3) * 2 + 1
        A = np.apply_along_axis(smooth1d, 0, A, window_len)
        A = np.apply_along_axis(smooth1d, 1, A, window_len)
        return A

    class BaseFilter:

        def get_pad(self, dpi):
            return 0

        def process_image(self, padded_src, dpi):
            raise NotImplementedError("Should be overridden by subclasses")

        def __call__(self, im, dpi):
            pad = self.get_pad(dpi)
            padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)],
                                "constant")
            tgt_image = self.process_image(padded_src, dpi)
            return tgt_image, -pad, -pad

    class OffsetFilter(BaseFilter):

        def __init__(self, offsets=(0, 0)):
            self.offsets = offsets

        def get_pad(self, dpi):
            return int(max(self.offsets) / 72 * dpi)

        def process_image(self, padded_src, dpi):
            ox, oy = self.offsets
            a1 = np.roll(padded_src, int(ox / 72 * dpi), axis=1)
            a2 = np.roll(a1, -int(oy / 72 * dpi), axis=0)
            return a2

    class GaussianFilter(BaseFilter):
        """Simple Gaussian filter."""

        def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
            self.sigma = sigma
            self.alpha = alpha
            self.color = color

        def get_pad(self, dpi):
            return int(self.sigma*3 / 72 * dpi)

        def process_image(self, padded_src, dpi):
            tgt_image = np.empty_like(padded_src)
            tgt_image[:, :, :3] = self.color
            tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha,
                                          self.sigma / 72 * dpi)
            return tgt_image

    class DropShadowFilter(BaseFilter):

        def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
            self.gauss_filter = GaussianFilter(sigma, alpha, color)
            self.offset_filter = OffsetFilter(offsets)

        def get_pad(self, dpi):
            return max(self.gauss_filter.get_pad(dpi),
                       self.offset_filter.get_pad(dpi))

        def process_image(self, padded_src, dpi):
            t1 = self.gauss_filter.process_image(padded_src, dpi)
            t2 = self.offset_filter.process_image(t1, dpi)
            return t2

    fig, ax = plt.subplots()

    # draw lines
    line1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], "bo-",
                     mec="b", mfc="w", lw=5, mew=3, ms=10, label="Line 1")
    line2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], "ro-",
                     mec="r", mfc="w", lw=5, mew=3, ms=10, label="Line 1")

    gauss = DropShadowFilter(4)

    for line in [line1, line2]:

        # draw shadows with same lines with slight offset.
        xx = line.get_xdata()
        yy = line.get_ydata()
        shadow, = ax.plot(xx, yy)
        shadow.update_from(line)

        # offset transform
        transform = mtransforms.offset_copy(
            line.get_transform(), fig, x=4.0, y=-6.0, units='points')
        shadow.set_transform(transform)

        # adjust zorder of the shadow lines so that it is drawn below the
        # original lines
        shadow.set_zorder(line.get_zorder() - 0.5)
        shadow.set_agg_filter(gauss)
        shadow.set_rasterized(True)  # to support mixed-mode renderers

    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def test_too_large_image():
    fig = plt.figure(figsize=(300, 2**25))
    buff = io.BytesIO()
    with pytest.raises(ValueError):
        fig.savefig(buff)


def test_chunksize():
    x = range(200)

    # Test without chunksize
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x))
    fig.canvas.draw()

    # Test with chunksize
    fig, ax = plt.subplots()
    rcParams['agg.path.chunksize'] = 105
    ax.plot(x, np.sin(x))
    fig.canvas.draw()


@pytest.mark.backend('Agg')
def test_jpeg_dpi():
    # Check that dpi is set correctly in jpg files.
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi=200)
    im = Image.open(buf)
    assert im.info['dpi'] == (200, 200)


def test_pil_kwargs_png():
    from PIL.PngImagePlugin import PngInfo
    buf = io.BytesIO()
    pnginfo = PngInfo()
    pnginfo.add_text("Software", "test")
    plt.figure().savefig(buf, format="png", pil_kwargs={"pnginfo": pnginfo})
    im = Image.open(buf)
    assert im.info["Software"] == "test"


def test_pil_kwargs_tiff():
    buf = io.BytesIO()
    pil_kwargs = {"description": "test image"}
    plt.figure().savefig(buf, format="tiff", pil_kwargs=pil_kwargs)
    im = Image.open(buf)
    tags = {TiffTags.TAGS_V2[k].name: v for k, v in im.tag_v2.items()}
    assert tags["ImageDescription"] == "test image"


@require_pillow_feature('WebP')
def test_pil_kwargs_webp():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf_small = io.BytesIO()
    pil_kwargs_low = {"quality": 1}
    plt.savefig(buf_small, format="webp", pil_kwargs=pil_kwargs_low)
    assert len(pil_kwargs_low) == 1
    buf_large = io.BytesIO()
    pil_kwargs_high = {"quality": 100}
    plt.savefig(buf_large, format="webp", pil_kwargs=pil_kwargs_high)
    assert len(pil_kwargs_high) == 1
    assert buf_large.getbuffer().nbytes > buf_small.getbuffer().nbytes


@require_pillow_feature('AVIF')
def test_pil_kwargs_avif():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf_small = io.BytesIO()
    pil_kwargs_low = {"quality": 1}
    plt.savefig(buf_small, format="avif", pil_kwargs=pil_kwargs_low)
    assert len(pil_kwargs_low) == 1
    buf_large = io.BytesIO()
    pil_kwargs_high = {"quality": 100}
    plt.savefig(buf_large, format="avif", pil_kwargs=pil_kwargs_high)
    assert len(pil_kwargs_high) == 1
    assert buf_large.getbuffer().nbytes > buf_small.getbuffer().nbytes


def test_gif_no_alpha():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="gif", transparent=False)
    im = Image.open(buf)
    assert im.mode == "P"
    assert im.info["transparency"] >= len(im.palette.colors)


def test_gif_alpha():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="gif", transparent=True)
    im = Image.open(buf)
    assert im.mode == "P"
    assert im.info["transparency"] < len(im.palette.colors)


@require_pillow_feature('WebP')
def test_webp_alpha():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="webp", transparent=True)
    im = Image.open(buf)
    assert im.mode == "RGBA"


@require_pillow_feature('AVIF')
def test_avif_alpha():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format="avif", transparent=True)
    im = Image.open(buf)
    assert im.mode == "RGBA"


def test_draw_path_collection_error_handling():
    fig, ax = plt.subplots()
    ax.scatter([1], [1]).set_paths(Path([(0, 1), (2, 3)]))
    with pytest.raises(TypeError):
        fig.canvas.draw()


def test_chunksize_fails(high_memory):
    # NOTE: This test covers multiple independent test scenarios in a single
    #       function, because each scenario uses ~2GB of memory and we don't
    #       want parallel test executors to accidentally run multiple of these
    #       at the same time.

    N = 100_000
    dpi = 500
    w = 5*dpi
    h = 6*dpi

    # make a Path that spans the whole w-h rectangle
    x = np.linspace(0, w, N)
    y = np.ones(N) * h
    y[::2] = 0
    path = Path(np.vstack((x, y)).T)
    # effectively disable path simplification (but leaving it "on")
    path.simplify_threshold = 0

    # setup the minimal GraphicsContext to draw a Path
    ra = RendererAgg(w, h, dpi)
    gc = ra.new_gc()
    gc.set_linewidth(1)
    gc.set_foreground('r')

    gc.set_hatch('/')
    with pytest.raises(OverflowError, match='cannot split hatched path'):
        ra.draw_path(gc, path, IdentityTransform())
    gc.set_hatch(None)

    with pytest.raises(OverflowError, match='cannot split filled path'):
        ra.draw_path(gc, path, IdentityTransform(), (1, 0, 0))

    # Set to zero to disable, currently defaults to 0, but let's be sure.
    with rc_context({'agg.path.chunksize': 0}):
        with pytest.raises(OverflowError, match='Please set'):
            ra.draw_path(gc, path, IdentityTransform())

    # Set big enough that we do not try to chunk.
    with rc_context({'agg.path.chunksize': 1_000_000}):
        with pytest.raises(OverflowError, match='Please reduce'):
            ra.draw_path(gc, path, IdentityTransform())

    # Small enough we will try to chunk, but big enough we will fail to render.
    with rc_context({'agg.path.chunksize': 90_000}):
        with pytest.raises(OverflowError, match='Please reduce'):
            ra.draw_path(gc, path, IdentityTransform())

    path.should_simplify = False
    with pytest.raises(OverflowError, match="should_simplify is False"):
        ra.draw_path(gc, path, IdentityTransform())


def test_non_tuple_rgbaface():
    # This passes rgbaFace as an ndarray to draw_path.
    fig = plt.figure()
    fig.add_subplot(projection="3d").scatter(
        [0, 1, 2], [0, 1, 2], path_effects=[patheffects.Stroke(linewidth=4)])
    fig.canvas.draw()


def test_rendered_height_floating_point_precision():
    fig = plt.figure(figsize=(1, 2.03), dpi=100)
    assert fig.bbox.height < 203  # due to floating-point precision
    fig.canvas.draw()
    assert fig.canvas.buffer_rgba().shape == (203, 100, 4)


@check_figures_equal()
def test_path_autosnap(fig_test, fig_ref):
    axt = fig_test.subplots()
    axr = fig_ref.subplots()

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    triangle1 = np.array([[0, 2], [1, 2], [0, 3]])
    triangle2 = np.array([[0, 4], [1, 4], [1, 3]])

    # A square should autosnap
    axt.add_patch(Polygon(square, snap=None))
    axr.add_patch(Polygon(square, snap=True))

    # A triangle with the diagonal as one of the line segments should not autosnap
    axt.add_patch(Polygon(triangle1, snap=None))
    axr.add_patch(Polygon(triangle1, snap=False))

    # A triangle with the diagonal as the closing line segment should not autosnap
    axt.add_patch(Polygon(triangle2, snap=None))
    axr.add_patch(Polygon(triangle2, snap=False))

    # A path of multiple polygons should autosnap if each polygon would autosnap
    multiple1 = Path.make_compound_path(Polygon(square + [[2, 0]]).get_path(),
                                        Polygon(square + [[2, 2]]).get_path())
    axt.add_patch(PathPatch(multiple1, edgecolor='none', snap=None))
    axr.add_patch(PathPatch(multiple1, edgecolor='none', snap=True))

    # A path of multiple polygons should not autosnap if any polygon would not autosnap
    multiple2 = Path.make_compound_path(Polygon(square + [[4, 0]]).get_path(),
                                        Polygon(triangle1 + [[4, 0]]).get_path())
    axt.add_patch(PathPatch(multiple2, edgecolor='none', snap=None))
    axr.add_patch(PathPatch(multiple2, edgecolor='none', snap=False))

    axt.autoscale_view()
    axr.autoscale_view()


def test_draw_text_images_bounds():
    # Positions must describe bitmaps inside the buffer, as the renderer indexes
    # into it without checking.
    fig = plt.figure(figsize=(1, 1), dpi=40)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gc = renderer.new_gc()
    buffer = np.zeros(16, dtype=np.uint8)

    for positions in [[[1 << 20, 4, 4, 5, 20]],  # Offset past the end.
                      [[0, 400, 400, 5, 39]],  # Larger than the buffer.
                      [[0, -4, 4, 5, 20]]]:  # Negative size.
        with pytest.raises(ValueError, match='outside of buffer'):
            renderer._renderer.draw_text_images(
                buffer, np.array(positions, dtype=np.intp), gc)

    renderer._renderer.draw_text_images(
        buffer, np.array([[0, 2, 2, 1, 3]], dtype=np.intp), gc)


@pytest.mark.parametrize("face", ["red", "C0", "0.5", "#ff0000", ("red", 0.5),
                                  (1, 0, 0), (1, 0, 0, 0.5)])
def test_draw_path_color_names(face):
    # The renderer accepts anything `.colors.to_rgba` does, not just tuples.
    renderer = RendererAgg(10, 10, 72)
    path = Path([(1, 1), (9, 1), (9, 9), (1, 9), (1, 1)], closed=True)
    renderer.draw_path(renderer.new_gc(), path, IdentityTransform(), face)
    expected = RendererAgg(10, 10, 72)
    expected.draw_path(expected.new_gc(), path, IdentityTransform(),
                       mcolors.to_rgba(face))
    assert np.asarray(renderer.buffer_rgba())[5, 5, 3] > 0
    np.testing.assert_array_equal(renderer.buffer_rgba(), expected.buffer_rgba())


@pytest.mark.parametrize("face", ["red", "C0", "0.5", "#ff0000", ("red", 0.5),
                                  (1, 0, 0), (1, 0, 0, 0.5)])
def test_draw_markers_color_names(face):
    # The renderer accepts anything `.colors.to_rgba` does, not just tuples.
    marker = Path([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)], closed=True)
    path = Path([(5, 5)])

    def draw(face):
        renderer = RendererAgg(10, 10, 72)
        renderer.draw_markers(renderer.new_gc(), marker, IdentityTransform(),
                              path, IdentityTransform(), face)
        return renderer.buffer_rgba()

    assert np.asarray(draw(face))[5, 5, 3] > 0
    np.testing.assert_array_equal(draw(face), draw(mcolors.to_rgba(face)))


_COLOR_ARRAY_INPUTS = [
    ["red", "C0"], "red", ["#ff0000"], [(1, 0, 0), (0, 0, 1)],
    [("red", 0.5)], np.array([[1, 0, 0]]), np.array([[1, 0, 0, 0.5]])]


@pytest.mark.parametrize("colors", _COLOR_ARRAY_INPUTS)
def test_draw_path_collection_color_names(colors):
    # The renderer accepts anything `.colors.to_rgba_array` does.
    path = Path([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)], closed=True)

    def draw(colors):
        renderer = RendererAgg(10, 10, 72)
        renderer.draw_path_collection(
            renderer.new_gc(), IdentityTransform(), [path], [],
            np.array([[3, 3], [7, 7]]), IdentityTransform(), colors, colors,
            [1], [(0, None)], [True], [None], None, hatchcolors=colors)
        return renderer.buffer_rgba()

    # (3, 3) is at row 6 of the buffer, as y is flipped.
    assert np.asarray(draw(colors))[6, 3, 3] > 0
    np.testing.assert_array_equal(draw(colors),
                                  draw(mcolors.to_rgba_array(colors)))


@pytest.mark.parametrize("colors", _COLOR_ARRAY_INPUTS)
@pytest.mark.parametrize("edgecolors", [None, "blue", ["blue"]])
def test_draw_quad_mesh_color_names(colors, edgecolors):
    coordinates = np.array([[[0, 0], [5, 0], [10, 0]],
                            [[0, 5], [5, 5], [10, 5]],
                            [[0, 10], [5, 10], [10, 10]]], float)

    def draw(colors, edgecolors):
        renderer = RendererAgg(10, 10, 72)
        renderer.draw_quad_mesh(
            renderer.new_gc(), IdentityTransform(), 2, 2, coordinates,
            np.zeros((0, 2)), IdentityTransform(), colors, True, edgecolors)
        return renderer.buffer_rgba()

    ec = None if edgecolors is None else mcolors.to_rgba_array(edgecolors)
    assert np.asarray(draw(colors, edgecolors))[2, 2, 3] > 0
    np.testing.assert_array_equal(
        draw(colors, edgecolors), draw(mcolors.to_rgba_array(colors), ec))


def test_draw_path_collection_default_hatchcolors():
    # *hatchcolors* defaults to None for third-party compatibility.
    renderer = RendererAgg(10, 10, 72)
    path = Path([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)], closed=True)
    renderer.draw_path_collection(
        renderer.new_gc(), IdentityTransform(), [path], [],
        np.array([[5, 5]]),
        IdentityTransform(), ["red"], ["black"], [1], [(0, None)], [True],
        [None], None)
    assert np.asarray(renderer.buffer_rgba())[5, 5, 3] > 0


def _draw_path_collection(*, gc_setup=None, **overrides):
    # Draw a 60x60 square at (50, 50), passing the arguments that
    # `.Collection.draw` would, overridden by *overrides*.
    renderer = RendererAgg(100, 100, 72)
    gc = renderer.new_gc()
    if gc_setup is not None:
        gc_setup(gc)
    path = Path([(-30, -30), (30, -30), (30, 30), (-30, 30), (-30, -30)],
                closed=True)
    kwargs = dict(
        master_transform=IdentityTransform(), paths=[path], all_transforms=[],
        offsets=np.array([[50, 50]]), offset_trans=IdentityTransform(),
        facecolors=["none"], edgecolors=["red"], linewidths=[4],
        linestyles=[(0, None)], antialiaseds=[True], urls=[None],
        offset_position=None)
    kwargs.update(overrides)
    renderer.draw_path_collection(gc, **kwargs)
    return np.asarray(renderer.buffer_rgba()).copy()


@pytest.mark.parametrize("name", ["facecolors", "edgecolors"])
def test_draw_path_collection_none_colors(name):
    # None is not a valid color array.  It must be a ValueError and not a
    # TypeError, as `.Collection.draw` reads the latter as the renderer not
    # supporting *hatchcolors*.
    with pytest.raises(ValueError, match=f"{name} must not be None"):
        _draw_path_collection(**{name: None})


@pytest.mark.parametrize("name", ["all_transforms", "offsets", "linestyles"])
def test_draw_path_collection_none_arrays(name):
    with pytest.raises(ValueError, match=f"{name} must not be None"):
        _draw_path_collection(**{name: None})


@pytest.mark.parametrize("name", ["facecolors", "edgecolors", "hatchcolors"])
@pytest.mark.parametrize("bad", [5, object(), ["red", None], "notacolor"])
def test_draw_path_collection_bad_colors_are_value_errors(name, bad):
    with pytest.raises(ValueError):
        _draw_path_collection(**{name: bad})


def test_draw_path_collection_empty_hatchcolors():
    # Omitted, None, and empty all use the hatch color of the context.
    def setup(gc):
        gc.set_hatch("/")
        gc.set_hatch_color((0, 1, 0, 1))

    expected = _draw_path_collection(gc_setup=setup)
    assert ((expected[..., 1] > expected[..., 0]) & (expected[..., 3] > 0)).any()
    for hatchcolors in [None, [], np.zeros((0, 4))]:
        np.testing.assert_array_equal(
            _draw_path_collection(gc_setup=setup, hatchcolors=hatchcolors),
            expected)
    # A given hatch color overrides the one of the context.
    blue = _draw_path_collection(gc_setup=setup, hatchcolors=["blue"])
    assert not np.array_equal(blue, expected)
    assert ((blue[..., 2] > blue[..., 0]) & (blue[..., 3] > 0)).any()


@pytest.mark.parametrize("transforms", [[], np.zeros((0, 3, 3)),
                                        [np.eye(3)], np.eye(3)[None]])
def test_draw_path_collection_transforms(transforms):
    expected = _draw_path_collection(all_transforms=np.zeros((0, 3, 3)))
    np.testing.assert_array_equal(
        _draw_path_collection(all_transforms=transforms), expected)


@pytest.mark.parametrize("offsets", [[[50, 50]], ((50, 50),), np.array([[50, 50]])])
def test_draw_path_collection_offsets(offsets):
    expected = _draw_path_collection(offsets=np.array([[50., 50.]]))
    np.testing.assert_array_equal(
        _draw_path_collection(offsets=offsets), expected)


def test_draw_path_collection_no_offsets():
    # Empty offsets (as a list) draw the path once, untranslated.
    np.testing.assert_array_equal(
        _draw_path_collection(offsets=[]),
        _draw_path_collection(offsets=np.zeros((0, 2))))


@pytest.mark.parametrize("linestyles", ["--", "dashed", ["--"], [(0, (4, 2))],
                                        (0, (4, 2)), [(0, [4, 2])]])
def test_draw_path_collection_linestyles(linestyles):
    with rc_context({"lines.dashed_pattern": [4, 2]}):
        expected = _draw_path_collection(linestyles=[(0, (4, 2))])
        solid = _draw_path_collection(linestyles=[(0, None)])
        assert not np.array_equal(expected, solid)
        np.testing.assert_array_equal(
            _draw_path_collection(linestyles=linestyles), expected)


@pytest.mark.parametrize("linestyles", ["bogus", 5, [(None, (1, 2))]])
def test_draw_path_collection_bad_linestyles(linestyles):
    with pytest.raises(ValueError):
        _draw_path_collection(linestyles=linestyles)


def test_draw_quad_mesh_none_colors():
    renderer = RendererAgg(10, 10, 72)
    coordinates = np.zeros((2, 2, 2))
    kwargs = dict(gc=renderer.new_gc(), master_transform=IdentityTransform(),
                  meshWidth=1, meshHeight=1, coordinates=coordinates,
                  offsets=[], offsetTrans=IdentityTransform(), antialiased=True)
    with pytest.raises(ValueError, match="facecolors must not be None"):
        renderer.draw_quad_mesh(facecolors=None, edgecolors=None, **kwargs)
    # None edgecolors mean to use the facecolors.
    renderer.draw_quad_mesh(facecolors=["red"], edgecolors=None, **kwargs)
