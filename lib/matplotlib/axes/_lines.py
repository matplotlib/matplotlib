"""
Lines and spans
"""
import numpy as np

from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib import collections as mcoll
from matplotlib.cbook import iterable
from ._base import _AxesBase


class LinesAndSpans(object):
    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        Add a horizontal line across the axis.

        Parameters
        ----------
        y : scalar, optional, default: 0
            y position in data coordinates of the horizontal line.

        xmin : scalar, optional, default: 0
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        xmax : scalar, optional, default: 1
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        Returns
        -------
        `~matplotlib.lines.Line2D`

        Notes
        -----
        kwargs are the same as kwargs to plot, and can be
        used to control the line properties.  e.g.,

        Examples
        --------

        * draw a thick red hline at 'y' = 0 that spans the xrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axhline(linewidth=4, color='r')

        * draw a default hline at 'y' = 1 that spans the xrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axhline(y=1)

        * draw a default hline at 'y' = .5 that spans the the middle half of
            the xrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axhline(y=.5, xmin=0.25, xmax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
        with the exception of 'transform':

        %(Line2D)s

        See also
        --------
        `axhspan` for example plot and source code
        """

        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axhline generates its own transform.")
        ymin, ymax = self.get_ybound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        self._process_unit_info(ydata=y, kwargs=kwargs)
        yy = self.convert_yunits(y)
        scaley = (yy < ymin) or (yy > ymax)

        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)
        l = mlines.Line2D([xmin, xmax], [y, y], transform=trans, **kwargs)
        self.add_line(l)
        self.autoscale_view(scalex=False, scaley=scaley)
        return l

    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a vertical line across the axes.

        Parameters
        ----------
        x : scalar, optional, default: 0
            y position in data coordinates of the vertical line.

        ymin : scalar, optional, default: 0
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        ymax : scalar, optional, default: 1
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        Returns
        -------
        `~matplotlib.lines.Line2D`

        Examples
        ---------
        * draw a thick red vline at *x* = 0 that spans the yrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axvline(linewidth=4, color='r')

        * draw a default vline at *x* = 1 that spans the yrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axvline(x=1)

        * draw a default vline at *x* = .5 that spans the the middle half of
            the yrange::

            >>> from matplotlib import pyplot as plt
            >>> plt.axvline(x=.5, ymin=0.25, ymax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
        with the exception of 'transform':

        %(Line2D)s

        See also
        --------

        `axhspan` for example plot and source code
        """

        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axvline generates its own transform.")
        xmin, xmax = self.get_xbound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        self._process_unit_info(xdata=x, kwargs=kwargs)
        xx = self.convert_xunits(x)
        scalex = (xx < xmin) or (xx > xmax)

        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        self.autoscale_view(scalex=scalex, scaley=False)
        return l

    def hlines(self, y, xmin, xmax, colors='k', linestyles='solid',
               label='', **kwargs):
        """
        Plot horizontal lines at each `y` from `xmin` to `xmax`.

        Parameters
        ----------
        y : scalar or sequence of scalar
            y-indexes where to plot the lines.

        xmin, xmax : scalar or 1D array_like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have same length.

        colors : array_like of colors, optional, default: 'k'

        linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

        label : string, optional, default: ''

        Returns
        -------
        lines : `~matplotlib.collections.LineCollection`

        Other parameters
        ----------------
        kwargs :  `~matplotlib.collections.LineCollection` properties.

        See also
        --------
        vlines : vertical lines

        Examples
        --------
        .. plot:: mpl_examples/pylab_examples/vline_hline_demo.py

        """

        # We do the conversion first since not all unitized data is uniform
        # process the unit information
        self._process_unit_info([xmin, xmax], y, kwargs=kwargs)
        y = self.convert_yunits(y)
        xmin = self.convert_xunits(xmin)
        xmax = self.convert_xunits(xmax)

        if not iterable(y):
            y = [y]
        if not iterable(xmin):
            xmin = [xmin]
        if not iterable(xmax):
            xmax = [xmax]

        y = np.asarray(y)
        xmin = np.asarray(xmin)
        xmax = np.asarray(xmax)

        if len(xmin) == 1:
            xmin = np.resize(xmin, y.shape)
        if len(xmax) == 1:
            xmax = np.resize(xmax, y.shape)

        if len(xmin) != len(y):
            raise ValueError('xmin and y are unequal sized sequences')
        if len(xmax) != len(y):
            raise ValueError('xmax and y are unequal sized sequences')

        verts = [((thisxmin, thisy), (thisxmax, thisy))
                 for thisxmin, thisxmax, thisy in zip(xmin, xmax, y)]
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        if len(y) > 0:
            minx = min(xmin.min(), xmax.min())
            maxx = max(xmin.max(), xmax.max())
            miny = y.min()
            maxy = y.max()

            corners = (minx, miny), (maxx, maxy)

            self.update_datalim(corners)
            self.autoscale_view()

        return coll

    def vlines(self, x, ymin, ymax, colors='k', linestyles='solid',
               label='', **kwargs):
        """
        Plot vertical lines at each `x` from `ymin` to `ymax`.

        Parameters
        ----------
        x : scalar or 1D array_like
            x-indexes where to plot the lines.

        xmin, xmax : scalar or 1D array_like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have same length.

        colors : array_like of colors, optional, default: 'k'

        linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

        label : string, optional, default: ''

        Returns
        -------
        lines : `~matplotlib.collections.LineCollection`

        Other parameters
        ----------------
        kwargs : `~matplotlib.collections.LineCollection` properties.

        See also
        --------
        hlines : horizontal lines

        Examples
        ---------
        .. plot:: mpl_examples/pylab_examples/vline_hline_demo.py

        """

        self._process_unit_info(xdata=x, ydata=[ymin, ymax], kwargs=kwargs)

        # We do the conversion first since not all unitized data is uniform
        x = self.convert_xunits(x)
        ymin = self.convert_yunits(ymin)
        ymax = self.convert_yunits(ymax)

        if not iterable(x):
            x = [x]
        if not iterable(ymin):
            ymin = [ymin]
        if not iterable(ymax):
            ymax = [ymax]

        x = np.asarray(x)
        ymin = np.asarray(ymin)
        ymax = np.asarray(ymax)
        if len(ymin) == 1:
            ymin = np.resize(ymin, x.shape)
        if len(ymax) == 1:
            ymax = np.resize(ymax, x.shape)

        if len(ymin) != len(x):
            raise ValueError('ymin and x are unequal sized sequences')
        if len(ymax) != len(x):
            raise ValueError('ymax and x are unequal sized sequences')

        Y = np.array([ymin, ymax]).T

        verts = [((thisx, thisymin), (thisx, thisymax))
                 for thisx, (thisymin, thisymax) in zip(x, Y)]
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        if len(x) > 0:
            minx = min(x)
            maxx = max(x)

            miny = min(min(ymin), min(ymax))
            maxy = max(max(ymin), max(ymax))

            corners = (minx, miny), (maxx, maxy)
            self.update_datalim(corners)
            self.autoscale_view()

        return coll

    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """
        Draws a horizontal span (rectangle) across the axis.

        Parameters
        ----------

        ymin, ymax : scalars
            Coordinates in data units of the horizontal span.

        xmin, xmax: scalars, optional, default: 0, 1
            Coordinates in relative axes coordinates. 0 is bottom, 0.5 is the
            middle and 1 is top. This always spans the xrange, regardless of
            the xlim settings (even if you change them with the `set_xlim`
            method).

        Returns
        -------
        polygon : `~matplotlib.patches.Polygon`

        Notes
        -----
        Valid kwargs are :class:`~matplotlib.patches.Polygon` properties:

        %(Polygon)s

        Examples
        --------

        - draws a gray rectangle from `y = 0.25-0.75` that spans the horizontal
        extent of the axes::

            >>> from matplotlib import pyplot as plt
            >>> plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

        - a more complex example:

        .. plot:: mpl_examples/pylab_examples/axhspan_demo.py

        """
        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

        # process the unit information
        self._process_unit_info([xmin, xmax], [ymin, ymax], kwargs=kwargs)

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits([xmin, xmax])
        ymin, ymax = self.convert_yunits([ymin, ymax])

        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        self.autoscale_view(scalex=False)
        return p

    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Draws a vertical span (rectangle) across the axis.

        Parameters
        ----------

        xmin, xmax : scalars
            Coordinates in data units of the vertical span.

        ymin, ymax: scalars, optional, default: 0, 1
            Coordinates in relative axes coordinates. 0 is bottom, 0.5 is the
            middle and 1 is top. This always spans the xrange, regardless of
            the xlim settings (even if you change them with the `set_xlim`
            method).

        Returns
        -------
        polygon : `~matplotlib.patches.Polygon`

        Notes
        -----
        Valid kwargs are :class:`~matplotlib.patches.Polygon` properties:

        %(Polygon)s

        Examples
        ---------
        - draws a vertical green translucent rectangle from x=1.25 to 1.55 that
        spans the yrange of the axes::

            >>> from matplotlib import pyplot as plt
            >>> plt.axvspan(1.25, 1.55, facecolor='g', alpha=0.5)
        """
        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)

        # process the unit information
        self._process_unit_info([xmin, xmax], [ymin, ymax], kwargs=kwargs)

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits([xmin, xmax])
        ymin, ymax = self.convert_yunits([ymin, ymax])

        verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        self.autoscale_view(scaley=False)
        return p
