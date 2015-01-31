from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import eq_

from matplotlib.transforms import Bbox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text
from matplotlib.backends.backend_agg import RendererAgg


@image_comparison(baseline_images=['font_styles'])
def test_font_styles():
    from matplotlib import _get_data_path
    data_path = _get_data_path()

    def find_matplotlib_font(**kw):
        prop = FontProperties(**kw)
        path = findfont(prop, directory=data_path)
        return FontProperties(fname=path)

    from matplotlib.font_manager import FontProperties, findfont
    warnings.filterwarnings(
        'ignore',
        ('findfont: Font family \[u?\'Foo\'\] not found. Falling back to .'),
        UserWarning,
        module='matplotlib.font_manager')

    plt.figure()
    ax = plt.subplot(1, 1, 1)

    normalFont = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        size=14)
    ax.annotate(
        "Normal Font",
        (0.1, 0.1),
        xycoords='axes fraction',
        fontproperties=normalFont)

    boldFont = find_matplotlib_font(
        family="Foo",
        style="normal",
        variant="normal",
        weight="bold",
        stretch=500,
        size=14)
    ax.annotate(
        "Bold Font",
        (0.1, 0.2),
        xycoords='axes fraction',
        fontproperties=boldFont)

    boldItemFont = find_matplotlib_font(
        family="sans serif",
        style="italic",
        variant="normal",
        weight=750,
        stretch=500,
        size=14)
    ax.annotate(
        "Bold Italic Font",
        (0.1, 0.3),
        xycoords='axes fraction',
        fontproperties=boldItemFont)

    lightFont = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=200,
        stretch=500,
        size=14)
    ax.annotate(
        "Light Font",
        (0.1, 0.4),
        xycoords='axes fraction',
        fontproperties=lightFont)

    condensedFont = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=500,
        stretch=100,
        size=14)
    ax.annotate(
        "Condensed Font",
        (0.1, 0.5),
        xycoords='axes fraction',
        fontproperties=condensedFont)

    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['multiline'])
def test_multiline():
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("multiline\ntext alignment")

    plt.text(
        0.2, 0.5, "TpTpTp\n$M$\nTpTpTp", size=20, ha="center", va="top")

    plt.text(
        0.5, 0.5, "TpTpTp\n$M^{M^{M^{M}}}$\nTpTpTp", size=20,
        ha="center", va="top")

    plt.text(
        0.8, 0.5, "TpTpTp\n$M_{q_{q_{q}}}$\nTpTpTp", size=20,
        ha="center", va="top")

    plt.xlim(0, 1)
    plt.ylim(0, 0.8)

    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['antialiased'], extensions=['png'])
def test_antialiasing():
    matplotlib.rcParams['text.antialiased'] = True

    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.75, "antialiased", horizontalalignment='center',
             verticalalignment='center')
    fig.text(0.5, 0.25, "$\sqrt{x}$", horizontalalignment='center',
             verticalalignment='center')
    # NOTE: We don't need to restore the rcParams here, because the
    # test cleanup will do it for us.  In fact, if we do it here, it
    # will turn antialiasing back off before the images are actually
    # rendered.


def test_afm_kerning():
    from matplotlib.afm import AFM
    from matplotlib.font_manager import findfont

    fn = findfont("Helvetica", fontext="afm")
    with open(fn, 'rb') as fh:
        afm = AFM(fh)
    assert afm.string_width_height('VAVAVAVAVAVA') == (7174.0, 718)


@image_comparison(baseline_images=['text_contains'], extensions=['png'])
def test_contains():
    import matplotlib.backend_bases as mbackend

    fig = plt.figure()
    ax = plt.axes()

    mevent = mbackend.MouseEvent(
        'button_press_event', fig.canvas, 0.5, 0.5, 1, None)

    xs = np.linspace(0.25, 0.75, 30)
    ys = np.linspace(0.25, 0.75, 30)
    xs, ys = np.meshgrid(xs, ys)

    txt = plt.text(
        0.48, 0.52, 'hello world', ha='center', fontsize=30, rotation=30)
    # uncomment to draw the text's bounding box
    # txt.set_bbox(dict(edgecolor='black', facecolor='none'))

    # draw the text. This is important, as the contains method can only work
    # when a renderer exists.
    plt.draw()

    for x, y in zip(xs.flat, ys.flat):
        mevent.x, mevent.y = plt.gca().transAxes.transform_point([x, y])
        contains, _ = txt.contains(mevent)
        color = 'yellow' if contains else 'red'

        # capture the viewLim, plot a point, and reset the viewLim
        vl = ax.viewLim.frozen()
        ax.plot(x, y, 'o', color=color)
        ax.viewLim.set(vl)


@image_comparison(baseline_images=['titles'])
def test_titles():
    # left and right side titles
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("left title", loc="left")
    ax.set_title("right title", loc="right")
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['text_alignment'])
def test_alignment():
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    x = 0.1
    for rotation in (0, 30):
        for alignment in ('top', 'bottom', 'baseline', 'center'):
            ax.text(
                x, 0.5, alignment + " Tj", va=alignment, rotation=rotation,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(
                x, 1.0, r'$\sum_{i=0}^{j}$', va=alignment, rotation=rotation)
            x += 0.1

    ax.plot([0, 1], [0.5, 0.5])
    ax.plot([0, 1], [1.0, 1.0])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['axes_titles'], extensions=['png'])
def test_axes_titles():
    # Related to issue #3327
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title('center', loc='center', fontsize=20, fontweight=700)
    ax.set_title('left', loc='left', fontsize=12, fontweight=400)
    ax.set_title('right', loc='right', fontsize=12, fontweight=400)


@cleanup
def test_set_position():
    fig, ax = plt.subplots()

    # test set_position
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    plt.draw()

    init_pos = ann.get_window_extent(fig.canvas.renderer)
    shift_val = 15
    ann.set_position((shift_val, shift_val))
    plt.draw()
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b

    # test xyann
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    plt.draw()

    init_pos = ann.get_window_extent(fig.canvas.renderer)
    shift_val = 15
    ann.xyann = (shift_val, shift_val)
    plt.draw()
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b


@image_comparison(baseline_images=['text_bboxclip'])
def test_bbox_clipping():
    plt.text(0.9, 0.2, 'Is bbox clipped?', backgroundcolor='r', clip_on=True)
    t = plt.text(0.9, 0.5, 'Is fancy bbox clipped?', clip_on=True)
    t.set_bbox({"boxstyle": "round, pad=0.1"})


@image_comparison(baseline_images=['annotation_negative_coords'],
                  extensions=['png'])
def test_annotation_negative_coords():
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    ax.annotate("+fpt", (15, 40), xycoords="figure points")
    ax.annotate("+fpx", (25, 30), xycoords="figure pixels")
    ax.annotate("+apt", (35, 20), xycoords="axes points")
    ax.annotate("+apx", (45, 10), xycoords="axes pixels")

    ax.annotate("-fpt", (-55, -40), xycoords="figure points")
    ax.annotate("-fpx", (-45, -30), xycoords="figure pixels")
    ax.annotate("-apt", (-35, -20), xycoords="axes points")
    ax.annotate("-apx", (-25, -10), xycoords="axes pixels")


@cleanup
def test_text_annotation_get_window_extent():
    figure = Figure(dpi=100)
    renderer = RendererAgg(200, 200, 100)

    # Only text annotation
    annotation = Annotation('test', xy=(0, 0))
    annotation.set_figure(figure)

    text = Text(text='test', x=0, y=0)
    text.set_figure(figure)

    bbox = annotation.get_window_extent(renderer=renderer)

    text_bbox = text.get_window_extent(renderer=renderer)
    eq_(bbox.width, text_bbox.width)
    eq_(bbox.height, text_bbox.height)

    _, _, d = renderer.get_text_width_height_descent(
        'text', annotation._fontproperties, ismath=False)
    _, _, lp_d = renderer.get_text_width_height_descent(
        'lp', annotation._fontproperties, ismath=False)
    below_line = max(d, lp_d)

    # These numbers are specific to the current implementation of Text
    points = bbox.get_points()
    eq_(points[0, 0], 0.0)
    eq_(points[1, 0], text_bbox.width)
    eq_(points[0, 1], -below_line)
    eq_(points[1, 1], text_bbox.height - below_line)


@cleanup
def test_text_with_arrow_annotation_get_window_extent():
    headwidth = 21
    fig, ax = plt.subplots(dpi=100)
    txt = ax.text(s='test', x=0, y=0)
    ann = ax.annotate('test',
        xy=(0.0, 50.0),
        xytext=(50.0, 50.0), xycoords='figure pixels',
        arrowprops={
            'facecolor': 'black', 'width': 2,
            'headwidth': headwidth, 'shrink': 0.0})

    plt.draw()
    renderer = fig.canvas.renderer
    # bounding box of text
    text_bbox = txt.get_window_extent(renderer=renderer)
    # bounding box of annotation (text + arrow)
    bbox = ann.get_window_extent(renderer=renderer)
    # bounding box of arrow
    arrow_bbox = ann.arrow.get_window_extent(renderer)
    # bounding box of annotation text
    ann_txt_bbox = Text.get_window_extent(ann)

    # make sure annotation with in 50 px wider than
    # just the text
    eq_(bbox.width, text_bbox.width + 50.0)
    # make sure the annotation text bounding box is same size
    # as the bounding box of the same string as a Text object
    eq_(ann_txt_bbox.height, text_bbox.height)
    eq_(ann_txt_bbox.width, text_bbox.width)
    # compute the expected bounding box of arrow + text
    expected_bbox = Bbox.union([ann_txt_bbox, arrow_bbox])
    assert_almost_equal(bbox.height, expected_bbox.height)


@cleanup
def test_arrow_annotation_get_window_extent():
    figure = Figure(dpi=100)
    figure.set_figwidth(2.0)
    figure.set_figheight(2.0)
    renderer = RendererAgg(200, 200, 100)

    # Text annotation with arrow
    annotation = Annotation(
        '', xy=(0.0, 50.0), xytext=(50.0, 50.0), xycoords='figure pixels',
        arrowprops={
            'facecolor': 'black', 'width': 8, 'headwidth': 10, 'shrink': 0.0})
    annotation.set_figure(figure)
    annotation.draw(renderer)

    bbox = annotation.get_window_extent()
    points = bbox.get_points()

    eq_(bbox.width, 50.0)
    assert_almost_equal(bbox.height, 10.0 / 0.72)
    eq_(points[0, 0], 0.0)
    eq_(points[0, 1], 50.0 - 5 / 0.72)


@cleanup
def test_empty_annotation_get_window_extent():
    figure = Figure(dpi=100)
    figure.set_figwidth(2.0)
    figure.set_figheight(2.0)
    renderer = RendererAgg(200, 200, 100)

    # Text annotation with arrow
    annotation = Annotation(
        '', xy=(0.0, 50.0), xytext=(0.0, 50.0), xycoords='figure pixels')
    annotation.set_figure(figure)
    annotation.draw(renderer)

    bbox = annotation.get_window_extent()
    points = bbox.get_points()

    eq_(points[0, 0], 0.0)
    eq_(points[1, 0], 0.0)
    eq_(points[1, 1], 50.0)
    eq_(points[0, 1], 50.0)
