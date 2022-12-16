import io

from bokeh.io import export_png, export_svg, show
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import gridplot
from bokeh.models import Label
from bokeh.palettes import Category10
from bokeh.plotting import figure
import numpy as np

from .bokeh_util import filled_to_bokeh, lines_to_bokeh


class BokehRenderer:
    """Utility renderer using Bokeh to render a grid of plots over the same (x, y) range.

    Args:
        nrows (int, optional): Number of rows of plots, default ``1``.
        ncols (int, optional): Number of columns of plots, default ``1``.
        figsize (tuple(float, float), optional): Figure size in inches (assuming 100 dpi), default
            ``(9, 9)``.
        show_frame (bool, optional): Whether to show frame and axes ticks, default ``True``.
        want_svg (bool, optional): Whether output is required in SVG format or not, default
            ``False``.

    Warning:
        :class:`~contourpy.util.bokeh_renderer.BokehRenderer`, unlike
        :class:`~contourpy.util.mpl_renderer.MplRenderer`, needs to be told in advance if output to
        SVG format will be required later, otherwise it will assume PNG output.
    """
    def __init__(self, nrows=1, ncols=1, figsize=(9, 9), show_frame=True, want_svg=False):
        self._want_svg = want_svg
        self._palette = Category10[10]

        total_size = 100*np.asarray(figsize)  # Assuming 100 dpi.

        nfigures = nrows*ncols
        self._figures = []
        backend = "svg" if self._want_svg else "canvas"
        for _ in range(nfigures):
            fig = figure(output_backend=backend)
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            self._figures.append(fig)
            if not show_frame:
                fig.outline_line_color = None
                fig.axis.visible = False

        self._layout = gridplot(
            self._figures, ncols=ncols, toolbar_location=None,
            width=total_size[0] // ncols, height=total_size[1] // nrows)

    def _convert_color(self, color):
        if isinstance(color, str) and color[0] == "C":
            index = int(color[1:])
            color = self._palette[index]
        return color

    def _get_figure(self, ax):
        if isinstance(ax, int):
            ax = self._figures[ax]
        return ax

    def _grid_as_2d(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        return x, y

    def filled(self, filled, fill_type, ax=0, color="C0", alpha=0.7):
        """Plot filled contours on a single plot.

        Args:
            filled (sequence of arrays): Filled contour data as returned by
                :func:`~contourpy.ContourGenerator.filled`.
            fill_type (FillType): Type of ``filled`` data, as returned by
                :attr:`~contourpy.ContourGenerator.fill_type`.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot with. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"C0"``.
            alpha (float, optional): Opacity to plot with, default ``0.7``.
        """
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        xs, ys = filled_to_bokeh(filled, fill_type)
        if len(xs) > 0:
            fig.multi_polygons(xs=[xs], ys=[ys], color=color, fill_alpha=alpha, line_width=0)

    def grid(self, x, y, ax=0, color="black", alpha=0.1, point_color=None, quad_as_tri_alpha=0):
        """Plot quad grid lines on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot grid lines, default ``"black"``.
            alpha (float, optional): Opacity to plot lines with, default ``0.1``.
            point_color (str, optional): Color to plot grid points or ``None`` if grid points
                should not be plotted, default ``None``.
            quad_as_tri_alpha (float, optional): Opacity to plot ``quad_as_tri`` grid, default
                ``0``.

        Colors may be a string color or the letter ``"C"`` followed by an integer in the range
        ``"C0"`` to ``"C9"`` to use a color from the ``Category10`` palette.

        Warning:
            ``quad_as_tri_alpha > 0`` plots all quads as though they are unmasked.
        """
        fig = self._get_figure(ax)
        x, y = self._grid_as_2d(x, y)
        xs = [row for row in x] + [row for row in x.T]
        ys = [row for row in y] + [row for row in y.T]
        kwargs = dict(line_color=color, alpha=alpha)
        fig.multi_line(xs, ys, **kwargs)
        if quad_as_tri_alpha > 0:
            # Assumes no quad mask.
            xmid = (0.25*(x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])).ravel()
            ymid = (0.25*(y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])).ravel()
            fig.multi_line(
                [row for row in np.stack((x[:-1, :-1].ravel(), xmid, x[1:, 1:].ravel()), axis=1)],
                [row for row in np.stack((y[:-1, :-1].ravel(), ymid, y[1:, 1:].ravel()), axis=1)],
                **kwargs)
            fig.multi_line(
                [row for row in np.stack((x[:-1, 1:].ravel(), xmid, x[1:, :-1].ravel()), axis=1)],
                [row for row in np.stack((y[:-1, 1:].ravel(), ymid, y[1:, :-1].ravel()), axis=1)],
                **kwargs)
        if point_color is not None:
            fig.circle(
                x=x.ravel(), y=y.ravel(), fill_color=color, line_color=None, alpha=alpha, size=8)

    def lines(self, lines, line_type, ax=0, color="C0", alpha=1.0, linewidth=1):
        """Plot contour lines on a single plot.

        Args:
            lines (sequence of arrays): Contour line data as returned by
                :func:`~contourpy.ContourGenerator.lines`.
            line_type (LineType): Type of ``lines`` data, as returned by
                :attr:`~contourpy.ContourGenerator.line_type`.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot lines. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"C0"``.
            alpha (float, optional): Opacity to plot lines with, default ``1.0``.
            linewidth (float, optional): Width of lines, default ``1``.

        Note:
            Assumes all lines are open line strips not closed line loops.
        """
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        xs, ys = lines_to_bokeh(lines, line_type)
        if len(xs) > 0:
            fig.multi_line(xs, ys, line_color=color, line_alpha=alpha, line_width=linewidth)

    def mask(self, x, y, z, ax=0, color="black"):
        """Plot masked out grid points as circles on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (masked array of shape (ny, nx): z-values.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Circle color, default ``"black"``.
        """
        mask = np.ma.getmask(z)
        if mask is np.ma.nomask:
            return
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        x, y = self._grid_as_2d(x, y)
        fig.circle(x[mask], y[mask], fill_color=color, size=10)

    def save(self, filename, transparent=False):
        """Save plots to SVG or PNG file.

        Args:
            filename (str): Filename to save to.
            transparent (bool, optional): Whether background should be transparent, default
                ``False``.

        Warning:
            To output to SVG file, ``want_svg=True`` must have been passed to the constructor.
        """
        if transparent:
            for fig in self._figures:
                fig.background_fill_color = None
                fig.border_fill_color = None

        if self._want_svg:
            export_svg(self._layout, filename=filename)
        else:
            export_png(self._layout, filename=filename)

    def save_to_buffer(self):
        """Save plots to an ``io.BytesIO`` buffer.

        Return:
            BytesIO: PNG image buffer.
        """
        image = get_screenshot_as_png(self._layout)
        buffer = io.BytesIO()
        image.save(buffer, "png")
        return buffer

    def show(self):
        """Show plots in web browser, in usual Bokeh manner.
        """
        show(self._layout)

    def title(self, title, ax=0, color=None):
        """Set the title of a single plot.

        Args:
            title (str): Title text.
            ax (int or Bokeh Figure, optional): Which plot to set the title of, default ``0``.
            color (str, optional): Color to set title. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``None`` which is ``black``.
        """
        fig = self._get_figure(ax)
        fig.title = title
        fig.title.align = "center"
        if color is not None:
            fig.title.text_color = self._convert_color(color)

    def z_values(self, x, y, z, ax=0, color="green", fmt=".1f", quad_as_tri=False):
        """Show ``z`` values on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (array-like of shape (ny, nx): z-values.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color of added text. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"green"``.
            fmt (str, optional): Format to display z-values, default ``".1f"``.
            quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centres
                of quads.

        Warning:
            ``quad_as_tri=True`` shows z-values for all quads, even if masked.
        """
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        kwargs = dict(text_color=color, text_align="center", text_baseline="middle")
        for j in range(ny):
            for i in range(nx):
                fig.add_layout(Label(x=x[j, i], y=y[j, i], text=f"{z[j, i]:{fmt}}", **kwargs))
        if quad_as_tri:
            for j in range(ny-1):
                for i in range(nx-1):
                    xx = np.mean(x[j:j+2, i:i+2])
                    yy = np.mean(y[j:j+2, i:i+2])
                    zz = np.mean(z[j:j+2, i:i+2])
                    fig.add_layout(Label(x=xx, y=yy, text=f"{zz:{fmt}}", **kwargs))
