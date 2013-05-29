from __future__ import print_function
import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
import matplotlib.pyplot as plt
import warnings
from nose.tools import with_setup


@image_comparison(baseline_images=['font_styles'])
def test_font_styles():
    from matplotlib import _get_data_path
    data_path = _get_data_path()

    def find_matplotlib_font(**kw):
        prop = FontProperties(**kw)
        path = findfont(prop, directory=data_path)
        return FontProperties(fname=path)

    from matplotlib.font_manager import FontProperties, findfont
    warnings.filterwarnings('ignore','findfont: Font family \[\'Foo\'\] '+ \
                            'not found. Falling back to .',
                            UserWarning,
                            module='matplotlib.font_manager')
    fig = plt.figure()
    ax = plt.subplot( 1, 1, 1 )

    normalFont = find_matplotlib_font( family = "sans-serif",
                                       style = "normal",
                                       variant = "normal",
                                       size = 14,
                                       )
    ax.annotate( "Normal Font", (0.1, 0.1), xycoords='axes fraction',
                  fontproperties = normalFont )

    boldFont = find_matplotlib_font( family = "Foo",
                                     style = "normal",
                                     variant = "normal",
                                     weight = "bold",
                                     stretch = 500,
                                     size = 14,
                                     )
    ax.annotate( "Bold Font", (0.1, 0.2), xycoords='axes fraction',
                  fontproperties = boldFont )

    boldItemFont = find_matplotlib_font( family = "sans serif",
                                         style = "italic",
                                         variant = "normal",
                                         weight = 750,
                                         stretch = 500,
                                         size = 14,
                                         )
    ax.annotate( "Bold Italic Font", (0.1, 0.3), xycoords='axes fraction',
                  fontproperties = boldItemFont )

    lightFont = find_matplotlib_font( family = "sans-serif",
                                      style = "normal",
                                      variant = "normal",
                                      weight = 200,
                                      stretch = 500,
                                      size = 14,
                                      )
    ax.annotate( "Light Font", (0.1, 0.4), xycoords='axes fraction',
                  fontproperties = lightFont )

    condensedFont = find_matplotlib_font( family = "sans-serif",
                                          style = "normal",
                                          variant = "normal",
                                          weight = 500,
                                          stretch = 100,
                                          size = 14,
                                          )
    ax.annotate( "Condensed Font", (0.1, 0.5), xycoords='axes fraction',
                  fontproperties = condensedFont )

    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['multiline'])
def test_multiline():
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("multiline\ntext alignment")

    plt.text(0.2, 0.5, "TpTpTp\n$M$\nTpTpTp", size=20,
             ha="center", va="top")

    plt.text(0.5, 0.5, "TpTpTp\n$M^{M^{M^{M}}}$\nTpTpTp", size=20,
             ha="center", va="top")

    plt.text(0.8, 0.5, "TpTpTp\n$M_{q_{q_{q}}}$\nTpTpTp", size=20,
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

    mevent = mbackend.MouseEvent('button_press_event', fig.canvas, 0.5,
                                 0.5, 1, None)

    xs = np.linspace(0.25, 0.75, 30)
    ys = np.linspace(0.25, 0.75, 30)
    xs, ys = np.meshgrid(xs, ys)

    txt = plt.text(0.48, 0.52, 'hello world', ha='center', fontsize=30,
                   rotation=30)
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
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("left title", loc="left")
    ax.set_title("right title", loc="right")
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(baseline_images=['text_alignment'])
def test_alignment():
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    x = 0.1
    for rotation in (0, 30):
        for alignment in ('top', 'bottom', 'baseline', 'center'):
            ax.text(x, 0.5, alignment + " Tj", va=alignment, rotation=rotation,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(x, 1.0, r'$\sum_{i=0}^{j}$', va=alignment, rotation=rotation)
            x += 0.1

    ax.plot([0, 1], [0.5, 0.5])
    ax.plot([0, 1], [1.0, 1.0])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])
