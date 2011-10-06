#!/usr/bin/env python
"""Classes to create ternary plots in matplotlib using a projection
"""
__author__ = "Kevin L. Davies"
__version__ = "2011/09/20"
__license__ = "BSD"
# The features and concepts were based on triangleplot.py by C. P. H. Lewis,
# 2008-2009, available at:
#     http://nature.berkeley.edu/~chlewis/Sourcecode.html
# The projection class and methods were adapted from
# custom_projection_example.py, available at:
#     http://matplotlib.sourceforge.net/examples/api/custom_projection_example.html

# **Todo:
#   1. Clean up the placement of the right axis label and ticklabels.  It may be
#      best to add the right axis as a special implementation of the y axis
#      rather than manually placing the gridlines, etc. as currently done.
#   2. Use the rcparams for r axis label and tick font size.
#   3. Allow data scaling (through self.total, set_data_ratio, or through
#      set_xlim and set_ylim).
#   4. Clean up the code and document it.
#   5. Email C. P. H. Lewis, post it to github for review, modify, and submit a
#      pull request.

# Types of plots:
#     Supported:
#         plot, scatter
#     *May* work as is, but should be tested and overridden with support for
#     b, l, r specification of data (todo):
#         contour, contourf, barbs, hexbin, imshow, fill, fill_between,
#         fill_betweenx, pcolor, pcolormesh, quiver, specgram, plotfile, spy,
#         tricontour, tricontourf, tripcolor
#     Not compatible and thus disabled:
#         acorr, bar, barh, boxplot, broken_barh, cohere, csd, hist, loglog,
#         pie, polar, psd, semilogx, semilogy, stem, xcorr

# Related methods:
#     Supported:
#         annotate, arrow, grid, legend, text, title, colorbar
#     New/unique:
#         set_rticks, **What others?
#     Should be unaffected (and thus compatible):
#         axes, axis, cla, clf, close, clabel, clim, delaxes, draw, figlegend,
#         figimage, figtext, figure, findobj, gca, gcf, gci, getp, hold, ioff,
#         ion, isinteractive, imread, imsave, ishold, matshow, rc, savefig,
#         subplot, subplots_adjust, subplot_tool, setp, show, suptitle
#     *May* work as is, but should be tested (todo):
#         box, locator_params, margins, plotfile, and table,
#         tick_params, ticklabel_format
#     Not applicable and thus disabled:
#         rgrids, thetagrids, twinx, twiny
#     **Need to fix, modify, and rename:
#         axhline, axhspan, axvline, axvspan, xlabel, ylabel, xlim, ylim,
#         xticks, yticks

# Colormap methods (should be also be compatible, but **should be tested):
#     autumn, bone, cool, copper, flag, gray, hot, hsv, jet, pink, prism,
#     spring, summer, winter, and spectral.

# Note: The existing methods for xlabel, xticks, etc. are mapped to the bottom
# axis, and those for ylabel, yticks, etc. are mapped to the left axis.
# Additional methods (**, **, **) have been added for the right axis.

# Known issues: The axes probably do not identify mouse events correctly.
# The offset_text_positions are probably not correct.

# **Need to test: Tick positions (inside/outside/both) on all 3 axes.

import numpy as np
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import matplotlib.text as mtext
import matplotlib.font_manager as font_manager
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines

from matplotlib.axes import _string_to_bool
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from matplotlib.cbook import iterable
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform, \
                                  IdentityTransform, Bbox

SQRT3 = np.sqrt(3.0)


class XTick(maxis.XTick):
    """Overwritten X tick methods
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
                rotation=-60, # New
                rotation_mode='anchor', # New
                )
            t.set_transform(trans)
            self._set_artist_props(t)
            return t


class YTick(maxis.YTick):
    """Overwritten Y tick methods
    """
#    def _get_gridline(self):
#        'Get the default line2D instance'
#        # x in axes coords, y in data coords
#        l = mlines.Line2D( xdata=(0,1), ydata=(0, 0),
#                    color=rcParams['grid.color'],
#                    linestyle=rcParams['grid.linestyle'],
#                    linewidth=rcParams['grid.linewidth'],
#                    )

#        #if self.label_position == 'left':
#        l.set_transform(self.axes.get_yaxis_transform(which='grid1'))
#        #else:
#        #    l.set_transform(self.axes.get_yaxis_transform(which='grid2'))
#        l.get_path()._interpolation_steps = GRIDLINE_INTERPOLATION_STEPS
#        self._set_artist_props(l)
#        return l


class XAxis(maxis.XAxis):
    """Customizations to the x-axis (in matplotlib.axis.XAxis)

    The x-axis is used for the bottom-axis (b-axis).
    """
    def set_label_position(self, position):
        raise NotImplementedError("Labels must remain on all three sides of "
                                  "the ternary plot.")
        return

    def _update_label_position(self, bboxes, bboxes2):
        """Determine the label's position from the bounding boxes of the
        ticklabels.
        """
        if not self._autolabelpos: return
        x, y = self.label.get_position()
        if self.label_position == 'bottom':
            if not len(bboxes):
                bottom = self.axes.bbox.ymin
                self.label.set_position((x, bottom -
                                        self.labelpad*self.figure.dpi / 72.0))
            else:
                bbox = mtransforms.Bbox.union(bboxes)
                bottom = bbox.y0
                # The label should also be shifted right by half the amount that
                # it is shifted downwards, so that it follows the imaginary,
                # extended gridlines at -60 deg.  However, since the y position
                # is in display coordinates and the x position is in data
                # coordinates, it is difficult to determine the proper shift.
                # The x shift of 0.06 (from a starting position of 0.5) seems
                # good by inspection.  **Replace this hack.
                self.label.set_position((0.56, bottom -
                                        self.labelpad*self.figure.dpi / 72.0))
        else:
            pass # The top label position is not used.

    def _get_tick(self, major):
        """Overwritten to use the XTick class from this file.
        """
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return XTick(self.axes, 0, '', major=major, **tick_kw)


class YAxis(maxis.YAxis):
    """Customizations to the y-axis (in matplotlib.axis.YAxis)

    The y-axis is used for the left-axis (l-axis).
    """
    def _get_label(self):
        # x in display coords (updated by _update_label_position)
        # y in axes coords
        label = mtext.Text(x=0, y=0.5,
            # todo: Get the label position.
            fontproperties=font_manager.FontProperties(
                               size=rcParams['axes.labelsize'],
                               weight=rcParams['axes.labelweight']),
            color=rcParams['axes.labelcolor'],
            verticalalignment='center',
            horizontalalignment='center',
            rotation_mode='anchor') # New
        label.set_transform(mtransforms.blended_transform_factory(
            mtransforms.IdentityTransform(), self.axes.transAxes) )
        self._set_artist_props(label)
        return label

    def _update_label_position(self, bboxes, bboxes2):
        """Determine the label's position from the bounding boxes of the
        ticklabels.
        """
        if not self._autolabelpos: return
        x, y = self.label.get_position()
        max_width = 0
        if self.label_position == 'left':
            if not len(bboxes):
                x = self.axes.bbox.xmin
            else:
                x = bboxes[0].x1 + 0.5*(bboxes[-1].x1 - bboxes[0].x1)
                for bbox in bboxes:
                    max_width = max(max_width, bbox.width)
            self.label.set_rotation(60)
            self.label.set_position((x - max_width +
                                     self.labelpad*self.figure.dpi/72.0, y))
        else:
            if not len(bboxes2):
                x = self.axes.bbox.xmax
            else:
                x = bboxes2[0].x0 + 0.5*(bboxes2[-1].x0 - bboxes2[0].x0)
                for bbox in bboxes2:
                    max_width = max(max_width, bbox.width)
            self.label.set_rotation(300)
            self.label.set_position((x + max_width +
                                     self.labelpad*self.figure.dpi/72.0, y))

    def set_label_position(self, position):
        """Set the label position (left or right)

        ACCEPTS: [ 'left' | 'right' ]
        """
        assert position == 'left' or position == 'right'
        self.label_position=position


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


class TernaryAxes(Axes):
    """A matplotlib projection for ternary axes

    Ternary plots are useful for plotting sets of three variables that each sum
    to the same value.  The variables are named b, l, and r (base, left, and
    right).

    See http://en.wikipedia.org/wiki/Ternary_plot.
    """
    name = 'ternary'

    class PreTernaryTransformBR(Transform):
        """This is an optional pre-processing step before the ternary transform;
        it maps *b* (base) and *r* (right) in ternary space to *b* (base) and
        *l* (left), also in ternary space.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, br):
            b = br[:, 0:1] # 0:1 rather than 1 in order to keep the data as a column
            r  = br[:, 1:2]
            #l = np.tile(self.total, r.shape) - b - r
            l = np.ones(r.shape) - b - r
            return np.concatenate((b, l), 1)

        def inverted(self):
            return TernaryAxes.InvertedPreTernaryTransformBR()
        inverted.__doc__ = Transform.inverted.__doc__


    class InvertedPreTernaryTransformBR(PreTernaryTransformBR):
        """This is the inverse of the optional pre-processing step; it maps *b*
        (base) and *l* (left) in ternary space to *b* (base) and *r* (right),
        also in ternary space.
        """
        # This transform is its own inverse.
        def inverted(self):
            return TernaryAxes.PreTernaryTransformBR()
        inverted.__doc__ = Transform.inverted.__doc__


    class TernaryTransform(Transform):
        """This is the ternary transform; it maps *b* (base) and *l* (left) into
        Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, bl):
            b = bl[:, 0:1]
            y  = bl[:, 1:2]
            x = b + y / 2.0
            return np.concatenate((x, y), 1)

        def inverted(self):
            return TernaryAxes.InvertedTernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__


    class InvertedTernaryTransform(Transform):
        """This is the inverse of the ternary transform; it maps *x* and *y* in
        Cartesian coordinate space back to *b* (base) and *l* (left).
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, xy):
            x = xy[:, 0:1]
            l = xy[:, 1:2]
            b = x - l / 2.0
            return np.concatenate((b, l), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__


    def resolve(self, b, l, r=None):
        """Resolve the over-specified system that arises when three variables
        and their sum is known.

        Return the first two variables (b and l) as singletons or numpy arrays
        as appropriate according to the inputs.

        If all three variables (b, l, and r) are available (i.e., are not None),
        then they are linearly scaled so that their sum is self.total.  If the
        relative error (i.e., the magnitude of the scaling factor minus one) is
        greater than self.tolerance, then the call fails an assertion.

        If b or l is None, the value of r and total is used to determine it.

        If two of the arguments (b, l, r) are None, then the call fails an
        assertion.  The arguments which are not None must have the same length,
        or the call fails another assertion.
        """
        def check(first, second):
            """Check if two variables are iterable, have the same length/shape,
            and are not None.

            If they are iterable, cast them as numpy array.  Return the
            variables and True if they are iterable (False if not).
            """
            assert first != None and second != None, \
                   "At least two of the values must be specified."
            if iterable(first) or iterable(r):
                first = np.array(first)
                second = np.array(second)
                assert first.shape == second.shape, \
                       ("Each of b, l, and r must have the same number of "
                        "elements if they are not None.")
                return first, second, True
            return first, second, False

        if b is None:
            l, r, is_iterable = check(l, r)
            if is_iterable:
                b = np.tile(self.total, l.shape) - l - r
            else:
                b = self.total - l - r
        elif l is None:
            b, r, is_iterable = check(b, r)
            if is_iterable:
                l = np.tile(self.total, b.shape) - b - r
            else:
                l = self.total - b - r
        elif r is None:
            b, l, is_iterable = check(b, l)
        else:
            if iterable(b) or iterable(l) or iterable(l):
                # At least one of b, l, and r is a list, tuple, array, or similar.
                b = np.array(b)
                l = np.array(l)
                r = np.array(r)
                assert b.shape == l.shape == b.shape, \
                       ("b, l, and r must have the same number of elements if "
                        "they are not None.")
                total = b + l + r
                factor = np.tile(self.total, b.shape) / total
                for i, relative_error in enumerate(np.absolute(1 - factor)):
                    # **Use verbose.
                    assert relative_error <= self.tolerance, \
                           ("The sum of entries with index %i is %f, but it "
                            "must be %f (within a relative tolerance of %f)."
                           ) %i%total[i] %self.total %self.tolerance
            else:
                total = b + l + r
                factor =self.total / total
                # **Use verbose.
                assert abs(1 - factor) <= self.tolerance, \
                       ("The sum of b, l, and r is %f, but it must be %f "
                        "(within a relative tolerance of %f).") %total \
                        %self.total %self.tolerance
            b *= factor
            l *= factor
        return b, l

    def plot(self, b=None, l=None, r=None, fmt=None, *args, **kwargs):
        """Plot lines and/or markers.
        """
        if type(r) is str:
            # Provide support for the third argument as a format string since
            # that is the way Axes.plot() works.
            return Axes.plot(self, b, l, r, *args, **kwargs)
        else:
            b, l = self.resolve(b, l, r)
            if fmt is None:
                return Axes.plot(self, b, l, *args, **kwargs)
            else:
                return Axes.plot(self, b, l, fmt, *args, **kwargs)

    def scatter(self, b=None, l=None, r=None, *args, **kwargs):
        """Scatter plot
        """
        b, l = self.resolve(b, l, r)
        return Axes.scatter(self, b, l, *args, **kwargs)

    def annotate(self, s, blr, blrtext=None, *args, **kwargs):
        """Add an annotation at location blr = (b, l, r) in data coordinates.
        """
        b, l = self.resolve(*blr)
        if blrtext is None:
            return Axes.annotate(self, s, (b, l), *args, **kwargs)
        else:
            btext, ltext = self.resolve(*blrtext)
            return Axes.annotate(self, s, (b, l), xytext=(btext, ltext), *args,
                                 **kwargs)

    def arrow(self, b=None, l=None, r=None, db=None, dl=None, dr=None,
              **kwargs):
        """Add an arrow to ternary axes from location b, l, r to (b + db,
        l + dl, r + dr) in data coordinates.
        """
        b2 = None if b is None or db is None else b + db
        l2 = None if l is None or dl is None else l + dl
        r2 = None if r is None or dr is None else r + dr
        b, l = self.resolve(b, l, r)
        b2, l2 = self.resolve(b2, l2, r2)
        return Axes.arrow(self, b, l, b2-b, l2-l, **kwargs)

    def text(self, b=None, l=None, r=None, *args, **kwargs):
        """Add text at location b, l, r in data coordinates.
        """
        if type(r) is str:
            # Provide support for the third argument as the text string since
            # that is the way Axes.text() works.
            return Axes.text(self, b, l, r, *args, **kwargs)
        else:
            b, l = self.resolve(b, l, r)
            return Axes.text(self, b, l, *args, **kwargs)

    # Prevent the user from applying nonlinear scales to either of the axes
    # since that would be confusing to the viewer.
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
    def rgrids(self, *args, **kwargs):
        raise NotImplementedError("Radial grids cannot be adjusted since polar "
                                  "plots are not supported by the ternary "
                                  "axes.")
        return
    def thetagrids(self, *args, **kwargs):
        raise NotImplementedError("Radial theta grids cannot be adjusted since "
                                  "polar plots are not supported by the "
                                  "ternary xes.")
        return
    def twinx(*args, **kwargs):
        raise NotImplementedError("Secondary y axes are not supported by the "
                                  "ternary axes.")
        return
    def twiny(*args, **kwargs):
        raise NotImplementedError("Secondary x axes are not supported by the "
                                  "ternary axes.")
        return

    def get_raxis(self):
        'Return the YAxis instance for the r-axis'
        return self.raxis

    def get_rgridlines(self):
        'Get the r grid lines as a list of Line2D instances'
        return cbook.silent_list('Line2D ygridline', self.raxis.get_gridlines())

    def get_rticklines(self):
        'Get the rtick lines as a list of Line2D instances'
        return cbook.silent_list('Line2D ytickline', self.raxis.get_ticklines())

    def set_xticks(self, *args, **kwargs):
        """Update the xticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.raxis.set_ticks(self, *args, **kwargs)

    def set_yticks(self, *args, **kwargs):
        """Update the yticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.raxis.set_ticks(self, *args, **kwargs)

    def set_rticks(self, *args, **kwargs):
        """Update the rticks, keeping the other ticks the same.
        """
        self.xaxis.set_ticks(self, *args, **kwargs)
        self.yaxis.set_ticks(self, *args, **kwargs)
        self.raxis.set_ticks(self, *args, **kwargs)

    # Modified from matplotlib.axes.Axes
    def grid(self, b=None, which='major', axis='both', **kwargs):
        """
        call signature::

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
            if self.raxis is not None:
                self.raxis.grid(b, which=which, **kwargs)

# **Add the rgrid back in.
        # Update the right axes grid and ticks.
        #for gridline in self.rgridlines:
        #    gridline.remove()
        #for ticklabel in self.rticklabels:
        #    ticklabel.remove()
        #self._draw_rgrid()
        #self._draw_rtick_labels()

    #def get_rticks(self):
    #    return self.rticks

    #def _draw_rtick_labels(self,  *args, **kwargs):
    #    # This is a hack.  **Clean it up.
    #    """Draw the r axis tick labels (matching the x tick labels).
    #    """
    #    offset = -0.04
    #    self.rticklabels=[]
    #    for label, tick in zip(self.get_xticklabels(), self.get_xticks()):
    #        label.set_rotation(-60)
    #        self.rticklabels.append(self.text(offset, 1.0-tick-offset,
    #                          str(tick), rotation=60, va='center', ha='center'))

    def set_rlabel(self, rlabel, fontdict=None, labelpad=None, **kwargs):
        """
        call signature::

          set_rlabel(rlabel, fontdict=None, labelpad=None, **kwargs)

        Set the label for the raxis

        *labelpad* is the spacing in points between the label and the r-axis

        Valid kwargs are Text properties:
        %(Text)s
        ACCEPTS: str

        .. seealso::

            :meth:`text`
                for information on how override and the optional args work
        """
        if labelpad is not None: self.raxis.labelpad = labelpad
        return self.raxis.set_label_text(rlabel, fontdict, **kwargs)

        #pass
    #    # This is a hack.  **Clean it up.
    #    offset = -0.12
    #    self.text(offset, 0.5-offset, rlabel, rotation=-60,
    #    va='center',
    #              ha='center')

    #def grid(self, b=None, **kwargs):
        # This is a hack.  **Clean it up.
        #Axes.grid(self, b=b, **kwargs)
        #if b:
        #    pass
        #    #self._draw_rgrid()
        #elif b == False:
        #    for gridline in self.rgridlines:
        #        gridline.set_visible(False)
        #else:
        #    # Toggle
        #    if len(self.rgridlines):
        #        visible = not self.rgridlines[0].get_visible()
        #        for gridline in self.rgridlines:
        #            gridline.set_visible(visible)

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
        shrink = kwargs.pop('shrink', 0.9)
        pad = kwargs.pop('pad', 0.1)
        return self.figure.colorbar(shrink=shrink, pad=pad, *args, **kwargs)

    def set_total(self, total):
        """Set the total of b, l, and r.
        """
        # This is a hack.  **Clean it up.
        self.total = total
        self._set_lim_and_transforms()
        self.set_xlim(0.0, self.total)
        self.set_ylim(0.0, self.total)
        self._update_transScale()

    def get_data_ratio(self):
        """Return the aspect ratio of the data itself.
        """
        return 1.0 # **Change this to self.total?

    def cla(self):
        """Override to set provide reasonable defaults.
        """
        # Call the base class.
        self.raxis = None # Placeholder until the r-axis is created
        Axes.cla(self)

        # Create the right axis (procedure modified from maplotlib.axes.twinx).
        if self._sharex is None:
            # Do this only once, or else there will be recursion.
            print self.get_position(True)
            ax2 = self.figure.add_axes(self.get_position(True), sharex=self,
                                       frameon=False, projection='ternary')
            ax2.xaxis.set_visible(False)
            self.yaxis.set_label_position('left')
            self.yaxis.set_offset_position('left')
            self.raxis = ax2.yaxis
            self.raxis.set_label_position('right')
            self.raxis.set_offset_position('right')

            # Do not display ticks (only gridlines, tick labels, and axis labels).
            self.xaxis.set_ticks_position('none')
            self.yaxis.set_ticks_position('none')
            self.raxis.set_ticks_position('none')

            # Turn off minor ticking altogether.
            self.xaxis.set_minor_locator(NullLocator())
            self.yaxis.set_minor_locator(NullLocator())
            self.raxis.set_minor_locator(NullLocator())

            # Vertical position of the title
            self.title.set_y(1.02)

            # Modify the padding between the y tick labels and the y axis label.
            # This value is from inpection, but it does seem to scale properly when
            # the figure is resized.
            self.yaxis.labelpad = -15

        self.grid(True)

        # Axes limits and scaling
        #self.set_xlim(0.0, self.total)
        #self.set_ylim(0.0, self.total)
        self.set_aspect(aspect='equal', adjustable='box-forced', anchor='C') # C is for center.

    def _set_lim_and_transforms(self):
        """Set up all the transforms for the data, text and grids when the plot
        is created.
        """
        # Three important coordinate spaces are defined here:
        #    1) Data space: The space of the data itself.
        #    2) Axes space: The unit rectangle (0, 0) to (1, 1)
        #       covering the entire plot area.
        #    3) Display space: The coordinates of the resulting image, often in
        #       pixels or dpi/inch.

        # 1) The core transformation from data space (b and l coordinates) into
        # Cartesian space defined in the TernaryTransform class.
        self.transProjection = self.TernaryTransform()

        # 2) The above has an output range that is not in the unit rectangle, so
        # scale and translate it.
        self.transAffine = Affine2D().scale(-1.8 / (SQRT3 * self.total),
                                            0.9 / self.total) \
                           + Affine2D().translate(0.5 + 0.9 / SQRT3, 0.05)
        # 3) This is the transformation from axes space to display space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Put these 3 transforms together---from data all the way to display
        # coordinates.  Using the '+' operator, these transforms are applied
        # "in order".  The transforms are automatically simplified, if possible,
        # by the underlying transformation framework.
        self.transData = self.transProjection + self.transAffine + self.transAxes

        # The main data transformation is set up.  Now deal with gridlines and
        # tick labels.

        # X-axis gridlines and ticklabels.  The input to these transforms are in
        # data coordinates in x and axis coordinates in y.  Therefore, the input
        # values will be in range (0, 0), (self.total, 1).  The goal of these
        # transforms is to go from that space to display space.  The tick labels
        # are offset 4 pixels.
        self._xaxis_transform = self.transData # Transform the axis itself.
        self._xaxis_text1_transform =  self.transData + Affine2D().translate(0, -4)
        self._xaxis_text2_transform = IdentityTransform() # For secondary x axes
                                                          # (required but not used)

        # Y- and R-axis gridlines and ticklabels.  The input to these transforms
        # are in axis coordinates in x and data coordinates in y. Therefore, the
        # input values will be in range (0, 0), (1, self.total).  These tick
        # labels are also offset 4 pixels.
        self._yaxis_transform = self.transData # Transform the y-axis itself.
        self._raxis_transform = self.PreTernaryTransformBR() + self._yaxis_transform
        self._yaxis_text1_transform = (Affine2D().scale(1, np.sqrt(2))
                                       + Affine2D().rotate_deg(45)
                                       + Affine2D().translate(1, 0)
                                       + self.transData
                                       + Affine2D().translate(-4, 0))
        self._yaxis_text2_transform = self.PreTernaryTransformBR() + self._yaxis_text1_transform

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
        if which == 'tick1' or (which == 'grid' and self._sharex is None):#self.xaxis.get_label_position == 'left'):
            return self._yaxis_transform
        else:
            return self._raxis_transform

    def get_yaxis_text1_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the y-axis
        tick labels (the left-axis).
        """
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pixelPad):
        """Return a tuple of the form (transform, valign, halign) for the
        secondary y-axis tick labels (the right-axis).
        """
        return self._yaxis_text2_transform, 'center', 'center'

    def _gen_axes_patch(self):
        """Return a patch for the background of the plot.

        The data and gridlines will be clipped to this shape.
        """
        vertices = np.array([[0.5 + 0.90 / SQRT3, 0.05],
                             [0.5, 0.95],
                             [0.5 - 0.90 / SQRT3, 0.05],
                             [0.5 + 0.90 / SQRT3, 0.05]])
#        vertices = np.array([[1, 0],
#                             [1, 1],
#                             [0, 1],
#                             [0, 0],
#                             [1, 0]])
        # The plot area must be shorter than 1.0 in order to leave room for the
        # title.
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
        # Don't register xaxis or yaxis with spines---as done in
        # Axes._init_axis()---until xaxis.cla() works.
        #self.spines['ternary'].register_axis(self.xaxis)
        #self.spines['ternary'].register_axis(self.yaxis)

        self._update_transScale()

    def __init__(self, *args, **kwargs):
        self.total = 1.0
        self.tolerance = 1e-12 # Relative error allow between the specified
                               # total and the sum of b, l, and r
                               # todo: Provide a way to change this.
        Axes.__init__(self, *args, **kwargs)
        self.cla()
