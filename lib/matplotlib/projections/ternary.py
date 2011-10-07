#!/usr/bin/env python
"""Classes to create a ternary plot using matplotlib projections
"""
__author__ = "Kevin L. Davies"
__version__ = "2011/10/7"
__license__ = "BSD"
# The features and concepts were based on triangleplot.py by C. P. H. Lewis,
# 2008-2009, available at:
#     http://nature.berkeley.edu/~chlewis/Sourcecode.html
# The projection class and methods were adapted from
# custom_projection_example.py, available at:
#     http://matplotlib.sourceforge.net/examples/api/custom_projection_example.html

# **Todo:
#   1. Fix set_xticks and set_yticks.
#   2. Use the rcparams for z axis label and tick font size.
#   3. Allow a, b, c sums other than 1.0 through data scaling (use self.total,
#      set_data_ratio, or set_xlim and set_ylim).
#   4. Clean up the code and document it.
#   5. Email C. P. H. Lewis, post it to github for review, modify, and submit a
#      pull request.

# Types of plots:
#     Supported:
#         plot, scatter, fill_between
#     *May* work as is, but should be tested:
#         contour, contourf, barbs, hexbin, imshow, fill, fill_betweenx, pcolor,
#         pcolormesh, quiver, specgram, plotfile, spy, tricontour, tricontourf,
#         tripcolor
#     Not compatible and thus disabled:
#         acorr, bar, barh, boxplot, broken_barh, cohere, csd, hist, loglog,
#         pie, polar, psd, semilogx, semilogy, stem, xcorr

# Related methods:
#     Supported:
#         annotate, arrow, colorbar, grid, legend, text, title, xlabel, ylabel
#     New/unique:
#         set_zlabel, set_alabel, set_blabel, set_clabel
#     Functionality changed:
#         twinx, twiny
#     Should be unaffected (and thus compatible):
#         axes, axis, cla, clf, close, clabel, clim, delaxes, draw, figlegend,
#         figimage, figtext, figure, findobj, gca, gcf, gci, getp, hold, ioff,
#         ion, isinteractive, imread, imsave, ishold, matshow, rc, savefig,
#         subplot, subplots_adjust, subplot_tool, setp, show, suptitle
#     *May* work as is, but **should be tested:
#         axhline, axhspan, axvline, axvspan, box, locator_params, margins,
#         plotfile, table, tick_params, ticklabel_format, xlim, xticks, ylim,
#         yticks
#     Not applicable and thus disabled:
#         zgrids, thetagrids

# Colormap methods (should be also be compatible, but **should be tested):
#     autumn, bone, cool, copper, flag, gray, hot, hsv, jet, pink, prism,
#     spring, summer, winter, and spectral.

# Note: The existing methods for xlabel, xticks, etc. are mapped to the right
# axis, and those for ylabel, yticks, etc. are mapped to the left axis. An
# additional method (set_zlabel) has been added for the bottom axis (labeled
# "z" for consistency.

# Known issues: The axes probably do not identify mouse events correctly.
# The offset_text_positions are probably not correct.  The tick angles and
# positions are not correct, but by default, ticks are not shown.  The zticks
# do not properly mirror the other ticks when updates are made.  The zgrid
# cannot be turned off (all grids are turned on by default).

import numpy as np
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import matplotlib.text as mtext
import matplotlib.font_manager as font_manager
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines

from matplotlib.axes import _string_to_bool
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform, \
                                  IdentityTransform, Bbox

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
GRIDLINE_INTERPOLATION_STEPS = 2

class YTick(maxis.YTick):
    """Customizations to the y-tick methods for ternary axes
    """
    def _get_text1(self):
            'Get the default Text instance'
            # the y loc is 3 points below the min of y axis
            # get the affine as an a,b,c,d,tx,ty list
            # x in data coords, y in axes coords
            #t =  mtext.Text(
            trans, vert, horiz = self._get_text1_transform()
            t = mtext.Text(
                x=0, y=0,
                fontproperties=font_manager.FontProperties(size=self._labelsize),
                color=self._labelcolor,
                verticalalignment=vert,
                horizontalalignment=horiz,
                rotation=120, # New
                rotation_mode='anchor', # New
                )
            t.set_transform(trans)
            self._set_artist_props(t)
            return t

class XAxis(maxis.XAxis):
    """Custom x-axis methods for ternary axes

    The x-axis is used for component A and is on the right side.
    """
    def set_label_position(self, position):
        raise NotImplementedError("Label positions are fixed in the ternary "
                                  "axes.")
        return

    def _get_label(self):
        # x and y are in display coordinates. **This is non-conventional.
        label = mtext.Text(x=0.5, y=0,
            fontproperties = font_manager.FontProperties(
                               size=rcParams['axes.labelsize'],
                               weight=rcParams['axes.labelweight']),
            color = rcParams['axes.labelcolor'],
            verticalalignment='bottom',
            horizontalalignment='center',
            rotation=-60, # Modified
            rotation_mode='anchor') # New

        self._set_artist_props(label)
        return label

    def _update_label_position(self, bboxes, bboxes2):
        """Determine the label's position from the bounding boxes of the
        ticklabels.
        """
        if not self._autolabelpos: return
        max_width = 0
        if not len(bboxes):
            x = self.axes.bbox.xmin
            y = (self.axes.bbox.ymax + self.axes.bbox.ymin)/2.0
        else:
            x = (bboxes[0].x0 + bboxes[-1].x0)/2.0
            y = (bboxes[0].y0 + bboxes[-1].y1)/2.0
            for bbox in bboxes:
                max_width = max(max_width, bbox.width)
        self.label.set_position((x + max_width +
                                 self.labelpad*self.figure.dpi/72.0, y))


class YAxis(maxis.YAxis):
    """Custom y-axis methods for ternary axes

    The y-axis is used for component B and is on the left side.
    """
    def set_label_position(self, position):
        raise NotImplementedError("Label positions are fixed in the ternary "
                                  "axes.")
        return

    def _get_label(self):
        # x and y are in display coordinates. **This is non-conventional.
        label = mtext.Text(x=0, y=0.5,
            # todo: Get the label position.
            fontproperties=font_manager.FontProperties(
                               size=rcParams['axes.labelsize'],
                               weight=rcParams['axes.labelweight']),
            color=rcParams['axes.labelcolor'],
            verticalalignment='center',
            horizontalalignment='center',
            rotation=60,
            rotation_mode='anchor') # New
        self._set_artist_props(label)
        return label

    def _update_label_position(self, bboxes, bboxes2):
        """Determine the label's position from the bounding boxes of the
        ticklabels.
        """
        if not self._autolabelpos: return

        # Y-axis labels
        max_width = 0
        max_height = 0
        if not len(bboxes):
            x = self.axes.bbox.xmin
            y = (self.axes.bbox.ymax + self.axes.bbox.ymin)/2.0
        else:
            # This shifts the label away from the center of the axis at
            # approximately 120 deg.
            y = (bboxes[0].y1 + bboxes[-1].y1)/2.0
            x = (bboxes[0].x0 + bboxes[-1].x0)/2.0
            for bbox in bboxes:
                max_width = max(max_width, bbox.width)
                max_height = max(max_height, bbox.height)
            space = np.sqrt(max_width**2 + max_height**2) # Length of tick labels
            space += self.labelpad*self.figure.dpi/72.0
        self.label.set_position((x - space/2.0,
                                 y + space*SQRT3/2.0))

        # Z-axis labels
        max_height = 0
        if not len(bboxes2):
            x = self.axes.bbox.xmin
            y = (self.axes.bbox.ymax + self.axes.bbox.ymin)/2.0
        else:
            # This shifts the label to its right somewhat, which is good,
            # although its center isn't exactly positioned from the center of
            # the axis at -60 deg as it should be.
            x = bboxes2[0].x1 + (bboxes2[-1].x0 - bboxes2[-1].x1)/2.0
            y = (bboxes2[0].y1 + bboxes2[-1].y1)/2.0
            for bbox in bboxes2:
                max_height = max(max_height, bbox.height)

    def _get_tick(self, major):
        """Overwritten to use the YTick class from this file.
        """
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return YTick(self.axes, 0, '', major=major, **tick_kw)


class Spine(mspines.Spine):
    """Customizations to matplotlib.spines.Spine"
    """
    @classmethod
    def triangular_spine(cls, axes, spine_type, **kwargs):
        """(staticmethod) Returns a linear :class:`Spine` for the ternary axes.

        Based on linear_spine()
        """
        if spine_type == 'bottom':
            path = Path([(0.0, 0.0), (1.0, 0.0)])
        elif spine_type == 'left':
            path = Path([(1.0, 0.0), (0.0, 1.0)])
        elif spine_type == 'right':
            path = Path([(0.0, 0.0), (0.0, 1.0)])
        else:
            raise ValueError("Unable to make path for spine " + spine_type)
        return cls(axes, spine_type, path, **kwargs)


class TernaryABAxes(Axes):
    """A matplotlib projection for ternary axes

    Ternary plots are useful for plotting sets of three variables that each sum
    to the same value.  For an introduction, see
    http://en.wikipedia.org/wiki/Ternary_plot for an introduction.

    Ternary plots may be rendered with clockwise- or counterclockwise-increasing
    axes.  This projection uses the counterclockwise convention.
    """
    name = 'ternaryab'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *a* (upper center component) and *b*
        (lower left component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, ab):
            a = ab[:, 0:1]
            b  = ab[:, 1:2]
            x = a + b - 1
            y = b
            return np.concatenate((x, y), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryABAxes.InvertedTernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedTernaryTransform(Transform):
        """This is the inverse of the non-separable part of the ternary
        transform (mapping *x* and *y* in Cartesian coordinate space back to *a*
        and *b*).
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            a = 1 + x - y
            b = y
            return np.concatenate((a, b), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryABAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    # Prevent the user from applying nonlinear scales to either of the axes
    # since that would be confusing to the viewer (the axes should have the same
    # scales, yet all three cannot be nonlinear at once).
    def set_xscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_xscale(self, *args, **kwargs)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_yscale(self, *args, **kwargs)

    # Disable changes to the data limits for now, but **support it later.
    def set_xlim(self, *args, **kwargs):
        pass
    def set_ylim(self, *args, **kwargs):
        pass

    # Disable interactive panning and zooming for now, but **support it later.
    def can_zoom(self):
        return False
    def start_pan(self, x, y, button):
        pass
    def end_pan(self):
        pass
    def drag_pan(self, button, key, x, y):
        pass

    # The following plots are not applicable for ternary axes.
    def acorr(self, *args, **kwargs):
        raise NotImplementedError("Autocorrelation plots are not supported by "
                                  "the ternary axes.")
        return
    def bar(self, *args, **kwargs):
        raise NotImplementedError("Bar plots are not supported by the ternary "
                                  + "axes.")
        return
    def barh(self, *args, **kwargs):
        raise NotImplementedError("Horizontal bar plots are not supported by "
                                  "the ternary axes.")
        return
    def boxplot(self, *args, **kwargs):
        raise NotImplementedError("Box plots are not supported by the ternary "
                                  "axes.")
        return
    def broken_barh(self, *args, **kwargs):
        raise NotImplementedError("Broken horizontal bar plots are not "
                                  "supported by the ternary axes.")
        return
    def cohere(self, *args, **kwargs):
        raise NotImplementedError("Coherance plots are not supported by the "
                                  "ternary axes.")
        return
    def csd(self, *args, **kwargs):
        raise NotImplementedError("Cross spectral density plots are not "
                                  "supported by the ternary axes.")
        return
    def plot_date(self, *args, **kwargs):
        raise NotImplementedError("Ternary plots cannot be labeled with dates.")
        return
    def hist(self, *args, **kwargs):
        raise NotImplementedError("Histograms are not supported by the ternary "
                                  "axes.")
        return
    def loglog(self, *args, **kwargs):
        raise NotImplementedError("Logarithmic plots are not supported by the "
                                  "ternary axes.")
        return
    def pie(self, *args, **kwargs):
        raise NotImplementedError("Pie charts are not supported by the ternary "
                                  "axes.")
        return
    def polar(self, *args, **kwargs):
        raise NotImplementedError("Polar plots are not supported by the "
                                  "ternary axes.")
        return
    def psd(self, *args, **kwargs):
        raise NotImplementedError("Power spectral density plots are not "
                                  "supported by the ternary axes.")
        return
    def semilogx(self, *args, **kwargs):
        raise NotImplementedError("Logarithmic plots are not supported by the "
                                  "ternary axes.")
        return
    def semilogy(self, *args, **kwargs):
        raise NotImplementedError("Logarithmic plots are not supported by the "
                                  "ternary axes.")
        return
    def stem(self, *args, **kwargs):
        raise NotImplementedError("Stem plots are not supported by the ternary "
                                  "axes.")
        return
    def xcorr(self, *args, **kwargs):
        raise NotImplementedError("Correlation plots are not supported by the "
                                  "the ternary axes.")
        return

    # The following methods are not applicable to ternary axes.
    def zgrids(self, *args, **kwargs):
        raise NotImplementedError("Radial grids cannot be adjusted since polar "
                                  "plots are not supported by the ternary "
                                  "axes.")
        return
    def thetagrids(self, *args, **kwargs):
        raise NotImplementedError("Radial theta grids cannot be adjusted since "
                                  "polar plots are not supported by the "
                                  "ternary xes.")
        return

    def twinx(self):
        """call signature::

          ax = twinx()

        Create a twin of TernaryAxesAB that uses (*b*, *c*) indexing rather than
        (*a*, *b*).  This is an "overloaded" version of the twiny() as defined
        by Axes.  There, twiny() creates an axes that shares the same y axis as
        the current axes.
        """
        axbc = self.figure.add_axes(self.get_position(True), sharex=self,
                                   projection='ternarybc', frameon=False)
        axbc.xaxis.set_visible(False)
        axbc.yaxis.set_visible(False)
        axbc.grid(False)
        return axbc

    def twiny(self):
        """call signature::

          ax = twiny()

        Create a twin of TernaryAxesAB that uses (*c*, *a*) indexing rather than
        (*a*, *b*).  This is an "overloaded" version of the twiny() as defined
        by Axes.  There, twiny() creates an axes that shares the same y axis as
        the current axes.
        """
        axca = self.figure.add_axes(self.get_position(True), sharey=self,
                                   projection='ternaryca', frameon=False)
        axca.xaxis.set_visible(False)
        axca.yaxis.set_visible(False)
        axca.grid(False)
        return axca

    def set_xticks(self, *args, **kwargs):
        """Update the xticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.zaxis._redraw_zticks(self, *args, **kwargs)

    def set_yticks(self, *args, **kwargs):
        """Update the yticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.zaxis._redraw_zticks(self, *args, **kwargs)

    def set_zticks(self, *args, **kwargs):
        """Update the zticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.zaxis._redraw_zticks(self, *args, **kwargs)

    # Modified from matplotlib.axes.Axes
    def grid(self, b=None, which='major', axis='both', **kwargs):
        """call signature::

           grid(self, b=None, which='major', axis='both', **kwargs)

        Set the axes grids on or off; *b* is a boolean.  (For MATLAB
        compatibility, *b* may also be a string, 'on' or 'off'.)

        If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
        *kwargs* are supplied, it is assumed that you want a grid and *b*
        is thus set to *True*.

        *which* can be 'major' (default), 'minor', or 'both' to control
        whether major tick grids, minor tick grids, or both are affected.

        *axis* can be 'both' (default), 'x', or 'y' to control which
        set of gridlines are drawn.

        *kawrgs* are used to set the grid line properties, eg::

           ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

        %(Line2D)s
        """
        # For now, the right axis' visibility simply follows the y-axis.
        if len(kwargs):
            b = True
        b = _string_to_bool(b)

        if axis == 'x' or  axis == 'both':
            self.xaxis.grid(b, which=which, **kwargs)
        if axis == 'y' or  axis == 'both':
            self.yaxis.grid(b, which=which, **kwargs)
            self._draw_zgrid(b)

    def _draw_zgrid(self, b, *args, **kwargs):
        """Draw the z grid (matching the x grid).
        """
        if self.zgridlines <> []:
            if b is None:
                b = self.yaxis.GridOn
            for l in self.zgridlines:
                l.set_visible(b) # **Why won't this turn the grid off?
                #if b:
                #    l.set_color(rcParams['grid.color'])
                #else:
                #    l.set_color('w')
                #    l.set_alpha(1)
                #    l.set_lw(0)
                #    self._set_artist_props(l)
        else:
            for x in self.get_xticks():
                l = mlines.Line2D(xdata=(0, 1-x), ydata=(1-x, 0),
                       color=rcParams['grid.color'],
                       linestyle=rcParams['grid.linestyle'],
                       linewidth=rcParams['grid.linewidth'],
                       )
                l.set_transform(self.axes.get_yaxis_transform(which='grid'))
                l.get_path()._interpolation_steps = GRIDLINE_INTERPOLATION_STEPS
                self._set_artist_props(l)
                self.zgridlines.append(l)
            self.figure.lines.extend(self.zgridlines)

    def _redraw_zgrid(self, *args, **kwargs):
        """Redraw the z grid (matching the x grid labels).
        """
        # This is a hack.  **Integrate it better.
        for gridline in self.zgridlines:
            gridline.remove()
        self._draw_ztick_labels(True, *args, **kwargs)

    def _draw_ztick_labels(self, *args, **kwargs):
        """Draw the z axis tick labels (matching the x tick labels).
        """
        # This is a hack.  **Integrate it better.
        self.zticklabels=[]
        trans, vert, horiz = self.get_yaxis_text2_transform(None)
        for label, tick in zip(self.get_xticklabels(), self.get_xticks()):
            txt = self.text(0, 1.0 - tick, str(tick),
                            rotation=240,
                            rotation_mode='anchor',
                            verticalalignment=vert,
                            horizontalalignment=horiz,
                            transform=trans)
            self.zticklabels.append(txt)

    def set_zlabel(self, zlabel, *args, **kwargs):
        """Set the z label, determining its position from the bounding boxes of
        the ticklabels.
        """
        # This is a hack.  **Clean it up.
        # bboxes = [zticklabel.clip_box for zticklabel in self.zticklabels]
        if self.zlabel is None:
        #    y = (-self.ticklabel_width
        #         - self.zlabelpad/2.0
        #         - 2*SQRT3)  # 2*SQRT3 is the y-projection of the 4-pixel offset.
        #    x = 0.5 - self.labelskew - self.zlabelpad*SQRT3/2.0
        #    self.zlabel = self.text(x, y, zticklabel,
        #                            verticalalignment='bottom',
        #                            horizontalalignment='center',
        #                            rotation=180, rotation_mode='anchor',
        #                            transform=self.zticklabels.get_transform())
            self.zlabel = self.text(0-self.zlabelpad, 0.5+self.zlabelpad,
                                    zlabel,
                                    verticalalignment='bottom',
                                    horizontalalignment='center',
                                    rotation=180, rotation_mode='anchor',
                                    transform=self.transData)
        else:
            self.zlabel.set_text(zlabel)

    def _remove_ztick_labels(self, *args, **kwargs):
        """Redraw the z axis tick labels (matching the x tick labels).
        """
        # This is a hack.  **Integrate it better.
        for ticklabel in self.zticklabels:
            ticklabel.remove()
        self._draw_ztick_labels(*args, **kwargs)

    def legend(self, *args, **kwargs):
        """Override the default legend location.
        """
        # The legend needs to be positioned outside the bounding box of the plot
        # area.  The normal specifications (e.g., legend(loc='upper right')) do
        # not do this.
        Axes.legend(self, bbox_to_anchor=kwargs.pop('bbox_to_anchor',
                                                    (1.05, 0.97)),
                    loc=kwargs.pop('loc', 'upper left'), *args, **kwargs)
        # This anchor position is by inspection.  It seems to give a good
        # horizontal position with default figure size and keeps the legend from
        # overlapped with the plot as the size of the figure is reduced.  The
        # top of the legend is vertically aligned with the top vertex of the
        # plot.

    def colorbar(self, *args, **kwargs):
        """Produce a colorbar with appropriate defaults for ternary plots.
        """
        shrink = kwargs.pop('shrink', self.height)
        pad = kwargs.pop('pad', 0.1)
        # **Need to shift the colorbar upwards according to self.elevation
        return self.figure.colorbar(shrink=shrink, pad=pad, *args, **kwargs)

    #def set_total(self, total):
    #    """Set the total of b, l, and r.
    #    """
    #    # This is a hack.  **Clean it up.
    #    self.total = total
    #    self._set_lim_and_transforms()
    #    self.set_xlim(0.0, self.total)
    #    self.set_ylim(0.0, self.total)
    #    self._update_transScale()

    #def get_data_ratio(self):
    #    """Return the aspect ratio of the data itself.
    #    """
    #    return 1.0 # **Change this to self.total?

    def cla(self):
        """Override to set provide reasonable defaults.
        """
        self.zgridlines = []
        self.zlabel = None

        # Call the base class.
        Axes.cla(self)

        # Create the right axis (procedure modified from maplotlib.axes.twinx).
        if self._sharex is None and self._sharey is None:
            self._draw_ztick_labels()
            self.grid(True)

        # Do not display ticks (only gridlines, tick labels, and axis labels).
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')

        # Turn off minor ticking altogether.
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())

        # Vertical position of the title
        self.title.set_y(1.02)

        # Modify the padding between the tick labels and the axis labels.
        # This value is from inpection, but it does seem to scale properly when
        # the figure is resized.
        self.xaxis.labelpad = 10 # In display units
        self.yaxis.labelpad = -15 # In display units
        self.zlabelpad = 0.1 # In data units

        # Axes limits and scaling
        #self.set_xlim(0.0, self.total)
        #self.set_ylim(0.0, self.total)

        # Spacing from the vertices (tips) to the tip labels (in data coords)
        self.tipoffset = 0.12

    def set_alabel(self, alabel, ha='center', va='center', rotation=0,
                   rotation_mode='anchor', **kwargs):
        """Add a tip label for component A.
        """
        # **This probably isn't very robust and should be improved.
        tipoffset = kwargs.pop('tipoffset', self.tipoffset)
        transform = kwargs.pop('transform', self.transData)
        self.atiplabel = self.text(1 + tipoffset, -tipoffset/2.0, alabel,
                                   ha=ha, va=va, rotation=rotation,
                                   rotation_mode=rotation_mode,
                                   transform=transform, **kwargs)
        return self.atiplabel

    def set_blabel(self, blabel, ha='center', va='center', rotation=120,
                   rotation_mode='anchor', **kwargs):
        """Add a tip label for component B.
        """
        # **This probably isn't very robust and should be improved.
        tipoffset = kwargs.pop('tipoffset', self.tipoffset)
        transform = kwargs.pop('transform', self.transData)
        self.btiplabel = self.text(-tipoffset/2.0, 1 + tipoffset, blabel,
                                   ha=ha, va=va, rotation=rotation,
                                   rotation_mode=rotation_mode,
                                   transform=transform, **kwargs)
        return self.btiplabel

    def set_clabel(self, clabel, ha='center', va='center', rotation=240,
                   rotation_mode='anchor', **kwargs):
        """Add a tip label for component C.
        """
        # **This probably isn't very robust and should be improved.
        tipoffset = kwargs.pop('tipoffset', self.tipoffset)
        transform = kwargs.pop('transform', self.transData)
        self.ctiplabel = self.text(-tipoffset/2.0, -tipoffset/2.0, clabel,
                                   ha=ha, va=va, rotation=rotation,
                                   rotation_mode=rotation_mode,
                                   transform=transform, **kwargs)
        return self.ctiplabel

    def _set_lim_and_transforms(self):
        """Set up all the transforms for the data, text, and grids when the plot
        is created.
        """
        # Three important coordinate spaces are defined here:
        #    1) Data space: The space of the data itself.
        #    2) Axes space: The unit rectangle (0, 0) to (1, 1)
        #       covering the entire plot area.
        #    3) Display space: The coordinates of the resulting image, often in
        #       pixels or dpi/inch.

        # 1) The core transformation from data space (a and b coordinates) into
        # Cartesian space defined in the TernaryTransform class.
        self.transProjection = self.TernaryTransform()

        # 2) The above has an output range that is not in the unit rectangle, so
        # scale and translate it.
        self.transAffine = (Affine2D().scale(-1, 1) + Affine2D().rotate_deg(45)
                           + Affine2D().scale(SQRT2/SQRT3, -SQRT2)
                           + Affine2D().scale(self.height)
                           + Affine2D().translate(0.5, self.height + self.elevation))
        # 3) This is the transformation from axes space to display space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Put these 3 transforms together -- from data all the way to display
        # coordinates.  Using the '+' operator, these transforms are applied
        # "in order".  The transforms are automatically simplified, if possible,
        # by the underlying transformation framework.
        self.transData = self.transProjection + self.transAffine + self.transAxes

        # The main data transformation is set up.  Now deal with gridlines and
        # tick labels.

        # X-axis gridlines and ticklabels.  The input to these transforms are in
        # data coordinates in x and axis coordinates in y.  Therefore, the input
        # values are in range (0, 0), (self.total, 1).  The goal of these
        # transforms is to go from that space to display space.
        self._xaxis_transform = self.transData
        self._xaxis_text1_transform = (self.transData
                                       + Affine2D().translate(4, 0)) # 4-pixel offset
        self._xaxis_text2_transform = IdentityTransform() # For secondary x axes
                                                          # (required but not used)

        # Y- and Z-axis gridlines and ticklabels.  The input to these transforms
        # are in axis coordinates in x and data coordinates in y.  Therefore,
        # the input values are in range (0, 0), (1, self.total).  These tick
        # labels are also offset 4 pixels.
        self._yaxis_transform = self.transData
        # **Can this be simplified?
        self._yaxis_text1_transform = (Affine2D().scale(-1, 1)
                                       + Affine2D().translate(1, 0)
                                       + self.transProjection
                                       + Affine2D().scale(-1, 1)
                                       + Affine2D().rotate_deg(45)
                                       + Affine2D().scale(SQRT2/SQRT3, -SQRT2)
                                       + Affine2D().scale(self.height)
                                       + Affine2D().rotate_deg(60)
                                       + Affine2D().translate(0.5, self.height + self.elevation)
                                       + self.transAxes
                                       + Affine2D().translate(-2, 2*SQRT3)) # 4-pixel offset
        self._yaxis_text2_transform = (self.transData
                                       + Affine2D().translate(-2, -2*SQRT3)) # 4-pixel offset

    def get_xaxis_transform(self, which='grid'):
        """Return the transformations for the x-axis grid and ticks.
        """
        assert which in ['tick1', 'tick2', 'grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the x-axis
        tick labels.
        """
        return self._xaxis_text1_transform, 'center', 'left'

    def get_xaxis_text2_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the x-axis
        tick labels.
        """
        # This is not used, but is provided for consistency.
        return self._xaxis_text2_transform, 'center', 'left'

    def get_yaxis_transform(self, which='grid'):
        """Return the transformations for the y-axis grid and ticks.
        """
        assert which in ['tick1', 'tick2', 'grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the y-axis
        tick labels (the left-axis).
        """
        return self._yaxis_text1_transform, 'center', 'left'

    def get_yaxis_text2_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the
        secondary y-axis tick labels (the right-axis).
        """
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        """Return a patch for the background of the plot.

        The data and gridlines will be clipped to this shape.
        """
        vertices = np.array([[0.5, self.elevation + self.height], # Point a=1 (upper center)
                             [0.5 - self.height/SQRT3,  self.elevation], # Point b=1 (lower left)
                             [0.5 + self.height/SQRT3, self.elevation], # Point c=1 (lower right)
                             [0.5, self.elevation]])
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        return PathPatch(Path(vertices, codes))

    def _gen_axes_spines(self):
        """Return a dict-set_ whose keys are spine names and values are Line2D
        or Patch instances.  Each element is used to draw a spine of the axes.
        """
        return {'ternary1':Spine.triangular_spine(self, 'bottom'),
                'ternary2':Spine.triangular_spine(self, 'left'),
                'ternary3':Spine.triangular_spine(self, 'right')}

    def _init_axis(self):
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)
        # Don't register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until xaxis.cla() works.
        #self.spines['ternary'].register_axis(self.xaxis)
        #self.spines['ternary'].register_axis(self.yaxis)
        self._update_transScale()

    def __init__(self, *args, **kwargs):
        # Sum of a, b, and c for this plot.
        #self.total = 1.0

        # Height of the ternary outline (in axis units)
        self.height = 0.8

        # Vertical distance between the y axis and the base of the ternary plot
        # (in axis units)
        self.elevation = 0.05

        Axes.__init__(self, *args, **kwargs)
        self.cla()
        self.set_aspect(aspect='equal', adjustable='box-forced', anchor='C') # C is for center.


class TernaryBCAxes(TernaryABAxes):
    name = 'ternarybc'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *b* (lower left component) and *c* (lower
        right component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        # **This could be factored into transAffine for efficiency.
        def transform(self, bc):
            b = bc[:, 0:1]
            c  = bc[:, 1:2]
            x = -c
            y = b
            return np.concatenate((x, y), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryBCAxes.InvertedTernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedTernaryTransform(Transform):
        """This is the inverse of the non-separable part of the ternary
        transform (mapping *x* and *y* in Cartesian coordinate space back to *b*
        and *c*).
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            b = y
            c = -x
            return np.concatenate((b, c), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryBCAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__


class TernaryCAAxes(TernaryABAxes):
    name = 'ternaryca'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *c* (lower right component) and *a* (upper
        center component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        # **Part of this could be factored into transAffine for efficiency.
        def transform(self, ca):
            c = ca[:, 0:1]
            a  = ca[:, 1:2]
            x = -c
            y = 1 - c - a
            return np.concatenate((x, y), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryCAAxes.InvertedTernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedTernaryTransform(Transform):
        """This is the inverse of the non-separable part of the ternary
        transform (mapping *x* and *y* in Cartesian coordinate space back to *c*
        and *a*).
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            c = -x
            a = 1 - c - y
            return np.concatenate((c, a), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryCAAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__
