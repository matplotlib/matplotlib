import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Rectangle
from matplotlib.testing._markers import needs_pgf_pdflatex
from matplotlib.testing.decorators import image_comparison

try:
    # Import the same cairo (pycairo or cairocffi) that is used by the backend
    from matplotlib.backends.backend_cairo import cairo
    cairo_version = cairo.cairo_version()
except ImportError:
    cairo_version = None


def plot_blend_mode_gallery(text=True, gouraud=True):
    N = 10
    data = np.arange(N**2).reshape((N, N)) % (N-1)

    fig, axs = plt.subplots(3, 8, figsize=(10, 5.5), dpi=80, layout="tight")
    axs = axs.flatten()
    fig.set_facecolor("none")

    blend_modes = ["normal", "multiply", "screen", "overlay",
                   "darken", "lighten", "color dodge", "color burn",
                   "hard light", "soft light", "difference", "exclusion",
                   "hue", "saturation", "color", "luminosity",
                   "knockout", "erase", "clear", "atop", "xor", "plus"]

    for ax in axs:
        ax.set_facecolor("none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.2)
        ax.set_axis_off()

    for i, blend_mode in enumerate(blend_modes):
        axs[i].imshow(data, cmap='Reds', alpha=0.75, extent=(0, 0.8, 0, 0.8))
        axs[i].imshow(data[::-1, :], cmap='Blues', alpha=0.75,
                      extent=(0.2, 1, 0.4, 1.2), blend_mode=blend_mode)
        if gouraud:
            axs[i].pcolormesh(*np.meshgrid(np.linspace(0.6, 0.9, 5),
                                           np.linspace(0.7, 1, 5)),
                              data[:5, :5], cmap='Spectral', alpha=0.75,
                              shading='gouraud', blend_mode=blend_mode)

        if text:
            axs[i].text(0.05, 0.15, "Test", weight="bold", color="c",
                        blend_mode=blend_mode)
            axs[i].text(0.35, 0.10, "Tilted", weight="bold", color="m", rotation=45,
                        blend_mode=blend_mode)

        axs[i].plot([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
                    [0.7, 0.8, 0.9, 1, 0.7, 0.8, 0.9, 1],
                    'p', markersize=15, markeredgecolor="orange",
                    markerfacecolor="purple", alpha=0.75, blend_mode=blend_mode)
        axs[i].plot([0, 1], [1.2, 0], color="y",
                    blend_mode=blend_mode)
        circ = Circle((.65, 0.5), .3, facecolor='g', alpha=0.5,
                      blend_mode=blend_mode, zorder=2)
        axs[i].add_artist(circ)

        rect = Rectangle((0, 1.2), 1, .3, facecolor='lightgray', clip_on=False)
        axs[i].add_artist(rect)

        if text:
            axs[i].set_title(blend_mode)


class ArtistGroup(Artist):
    def __init__(self, artist_list, *, group_blend_mode=None, group_alpha=1,
                 knockout=False):
        self._artist_list = artist_list
        self._group_blend_mode = group_blend_mode
        self._group_alpha = group_alpha
        self._knockout = knockout
        super().__init__()

    def draw(self, renderer):
        renderer.open_blend_group(self._group_blend_mode, alpha=self._group_alpha,
                                  knockout=self._knockout)
        for a in self._artist_list:
            if not a.is_transform_set():
                a.set_transform(self.get_transform())
            a.draw(renderer)
        renderer.close_blend_group()


def plot_blend_group_types():
    # Rows: top row is non-isolated, bottom row is isolated
    # Columns: left column is non-knockout, right column is knockout
    fig, axs = plt.subplots(2, 2, figsize=(3, 3), dpi=80, layout='constrained')

    for i, group_blend_mode in enumerate([None, "normal"]):
        for j, knockout in enumerate([False, True]):
            axs[i, j].set_xlim(-1, 1)
            axs[i, j].set_ylim(-1, 1)
            axs[i, j].set_aspect("equal")
            axs[i, j].set_axis_off()

            axs[i, j].imshow(np.arange(20*20).reshape((20, 20)) % 19,
                             cmap='Spectral', extent=[-1, 1, -1, 1])

            left = Circle((-0.25, 0), 0.6, fc='y', alpha=0.75, blend_mode='multiply')
            right = Circle((0.25, 0), 0.6, fc='g', alpha=0.75, blend_mode='multiply')

            both = ArtistGroup([left, right], group_blend_mode=group_blend_mode,
                               knockout=knockout)
            axs[i, j].add_artist(both)


@image_comparison(['blend_modes_agg.png'], style='mpl20')
def test_blend_modes_agg():
    plot_blend_mode_gallery()


@pytest.mark.backend('cairo')
@image_comparison(['blend_modes_cairo.png'], style='mpl20',
                  tol=3 if cairo_version is not None and cairo_version < 11804 else 0)
def test_blend_modes_cairo():
    # The test image used cairo 1.18.4, so loosen the tolerance for older cairo
    # Disable text because text rendering varies too much with environment
    plot_blend_mode_gallery(text=False)


@image_comparison(['blend_modes_svg.svg'], style='mpl20')
def test_blend_modes_svg():
    # Disable the Gouraud component because its implementation increases the image file
    # size by an order of magnitude, plus the implementation is actually not supported
    # by typical SVG viewers
    plot_blend_mode_gallery(gouraud=False)


@image_comparison(['blend_modes_pdf.pdf'], style='mpl20')
def test_blend_modes_pdf():
    plot_blend_mode_gallery()


@needs_pgf_pdflatex
@pytest.mark.backend('pgf')
@image_comparison(['blend_modes_pgf.pdf'], style='mpl20')
def test_blend_modes_pgf():
    # Disable the Gouraud component because it is not supported by the PGF backend
    plot_blend_mode_gallery(gouraud=False)


@image_comparison(['blend_groups_agg.png'], style='mpl20')
def test_blend_groups_agg():
    plot_blend_group_types()


@pytest.mark.backend('cairo')
@image_comparison(['blend_groups_cairo.png'], style='mpl20')
def test_blend_groups_cairo():
    plot_blend_group_types()


@image_comparison(['blend_groups_svg.svg'], style='mpl20')
def test_blend_groups_svg():
    plot_blend_group_types()


@image_comparison(['blend_groups_pdf.pdf'], style='mpl20')
def test_blend_groups_pdf():
    plot_blend_group_types()


@needs_pgf_pdflatex
@pytest.mark.backend('pgf')
@image_comparison(['blend_groups_pgf.pdf'], style='mpl20')
def test_blend_groups_pgf():
    plot_blend_group_types()
