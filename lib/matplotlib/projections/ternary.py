#!/usr/bin/env python
"""matplotlib projections for ternary axes
"""
__author__ = "Kevin L. Davies"
__version__ = "2011/10/8"
__license__ = "BSD"
# The features and concepts were based on triangleplot.py by C. P. H. Lewis,
# 2008-2009, available at:
#     http://nature.berkeley.edu/~chlewis/Sourcecode.html
# The projection class and methods were adapted from
# custom_projection_example.py, available at:
#     http://matplotlib.sourceforge.net/examples/api/custom_projection_example.html

# **Todo:
#   1. Fix the position and angles of xlabel, xticks, xticklabels.
#   2. Clean up the code and document it (in ReST format; give a summary of
#      features/changes relative to previous work).
#   3. Post it to github for review, modify, and submit a pull request.

# **Nice to have:
#   1. Allow a, b, c sums other than 1.0 through data scaling (using self.total,
#      set_data_ratio, or set_xlim?).
#   2. Add automatic offsetting/padding of tiplabels.

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
#         annotate, arrow, colorbar, grid, legend, text, title, xlabel
#     New/unique:
#         set_tiplabel
#     Functionality changed:
#         twinx
#     Should be unaffected (and thus compatible):
#         axes, axis, cla, clf, close, clabel, clim, delaxes, draw, figlegend,
#         figimage, figtext, figure, findobj, gca, gcf, gci, getp, hold, ioff,
#         ion, isinteractive, imread, imsave, ishold, matshow, rc, savefig,
#         subplot, subplots_adjust, subplot_tool, setp, show, suptitle
#     *May* work as is, but **should be tested:
#         axhline, axhspan, axvline, axvspan, box, locator_params, margins,
#         plotfile, table, tick_params, ticklabel_format, xlim, xticks, ylim,
#         yticks
#     Not applicable:
#         ylabel
#     Not applicable and thus disabled:
#         zgrids, thetagrids

# Colormap methods (should be also be compatible, but **should be tested):
#     autumn, bone, cool, copper, flag, gray, hot, hsv, jet, pink, prism,
#     spring, summer, winter, and spectral.

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


class XTick(maxis.XTick):
    """Customizations to the x-tick methods for ternary axes
    """
    def _get_text1(self):
        'Get the default Text instance'
        # the y loc is 3 points below the min of y axis
        # get the affine as an a,b,c,d,tx,ty list
        # x in data coords, y in axes coords
        trans, vert, horiz = self._get_text1_transform()
        t = mtext.Text(
            x=0, y=0,
            fontproperties=font_manager.FontProperties(size=self._labelsize),
            color=self._labelcolor,
            verticalalignment=vert,
            horizontalalignment=horiz,
            rotation=self.axes.angle, # New
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
            rotation=self.axes.angle - 60, # Modified
            rotation_mode='anchor') # New

        self._set_artist_props(label)
        return label

    def _update_label_position(self, bboxes, bboxes2):
        """Determine the label's position from the bounding boxes of the
        ticklabels.
        """
        #x,y = self.label.get_position()
        #x = (self.axes.bbox.xmin + self.axes.bbox.xmax)/2.0
        #y = (self.axes.bbox.ymin + self.axes.bbox.ymax)/2.0
        # **Fix this:

        if len(bboxes) and self._autolabelpos:
            # This is a hack to find the center of the x axis.  **Clean it up.
            if self.axes.angle == 0: # Axes c, a (label on right)
                x = (bboxes[0].x0 + bboxes[-1].x0)/2.0
                y = (bboxes[0].y1 + bboxes[-1].y0)/2.0
            elif self.axes.angle == 120: # Axes a, b (label on left)
                x = (bboxes[0].x1 + bboxes[-1].x1)/2.0
                y = (bboxes[0].y0 + bboxes[-1].y0)/2.0
            else: # Axes b, c (label on bottom)
                x = (bboxes[0].x1 + bboxes[-1].x1)/2.0
                y = (bboxes[0].y1 + bboxes[-1].y1)/2.0
            max_width = 0
            max_height = 0
            for bbox in bboxes:
                max_width = max(max_width, bbox.width)
                max_height = max(max_height, bbox.height)
            space = np.sqrt(max_width**2 + max_height**2) # Maximum possible length of tick labels
            space += self.labelpad*self.figure.dpi/72.0
            angle = np.radians(self.axes.angle) # Offset angle in radians
            self.label.set_position((x + np.cos(angle)*space,
                                     y + np.sin(angle)*space))

    def _get_tick(self, major):
        """Overwritten to use the XTick class from this file.
        """
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return XTick(self.axes, 0, '', major=major, **tick_kw)


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

    def _gen_transProjection(self):
        """Return the projection transformation.

        This is a method so that it can return an affine transformation (the
        identity transformation for the *a*, *b* axes) or a non-affine
        transformation (for the other two axes).
        """
        return IdentityTransform()

    def _gen_transAffinePart1(self):
        """This is the part of the affine transformation that is unique to the
        *a*, *b* ternary axes.
        """
        return Affine2D().rotate_deg(225) + Affine2D().scale(1/SQRT3, 1)

    def get_angle(self):
        """Return the offset angle [deg] of these axes relative to the a, b axes.
        """
        return 120

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

    def twinx(self, projection):
        """call signature::

          ax = twinx()

        Create a twin of the current ternary axes with a different projection,
        so that all of the indexing schemes can be used -- (*b*, *c*),
        (*a*, *b*), and (*c*, *a*).
        """
        return self.figure.add_axes(self.get_position(True), sharex=self,
                                    projection=projection, frameon=False)

    def legend(self, *args, **kwargs):
        """Override the default legend location.
        """
        # The legend needs to be positioned outside the bounding box of the plot
        # area.  The normal specifications (e.g., legend(loc='upper right')) do
        # not do this.
        loc=kwargs.pop('loc', 'upper left')
        borderaxespad=kwargs.pop('borderaxespad', 0)
        bbox_to_anchor=kwargs.pop('bbox_to_anchor', (0.5 + self.height/SQRT3, self.elevation + self.height))
        Axes.legend(self, loc=loc, borderaxespad=borderaxespad,
                    bbox_to_anchor=bbox_to_anchor, *args, **kwargs)
        # This anchor position is by inspection.  It seems to give a good
        # horizontal position with default figure size and keeps the legend from
        # overlapped with the plot as the size of the figure is reduced.  The
        # top of the legend is vertically aligned with the top vertex of the
        # plot.  **Update the position.

    def colorbar(self, *args, **kwargs):
        """Produce a colorbar with appropriate defaults for ternary plots.
        """
        pad = kwargs.pop('pad', 0.1)
        shrink = kwargs.pop('shrink', 1.0)
        fraction = kwargs.pop('fraction', 0.04)
        # This is a hack and the alignment isnt uite right. **Clean it up.
        scaley = rcParams['figure.subplot.top'] - rcParams['figure.subplot.bottom']
        cax = self.figure.add_axes([0.74 + pad,
                                    rcParams['figure.subplot.bottom'] + self.elevation,
                                    fraction,
                                    self.height*scaley*shrink - 0.005])
        return self.figure.colorbar(cax=cax, *args, **kwargs)
#        return self.figure.colorbar(shrink=shrink, pad=pad, *args, **kwargs)

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
        # Call the base class.
        Axes.cla(self)
        self.grid(True)

        # Only the x-axis is shown, but there are 3 of them once all of
        # projections are included.
        self.yaxis.set_visible(False)

        # Adjust the number of ticks shown.
        self.set_xticks(np.linspace(0, 1, 5))

        # Do not display ticks (only gridlines, tick labels, and axis labels).
        self.xaxis.set_ticks_position('none')

        # Turn off minor ticking altogether.
        self.xaxis.set_minor_locator(NullLocator())

        # Vertical position of the title
        self.title.set_y(1.02)

        # Modify the padding between the tick labels and the axis labels.
        self.xaxis.labelpad = 10 # In display units

        # Axes limits and scaling
        #self.set_xlim(0.0, self.total)

        # Spacing from the vertices (tips) to the tip labels (in data coords)
        self.tipoffset = 0.14

    def set_tiplabel(self, tiplabel, ha='center', va='center',
                   rotation_mode='anchor', **kwargs):
        """Add a tip label for the first component.
        """
        tipoffset = kwargs.pop('tipoffset', self.tipoffset)
        rotation = kwargs.pop('rotation', self.angle)
        self.tiplabel = self.text(1 + tipoffset, -tipoffset/2.0, tiplabel,
                                   ha=ha, va=va, rotation=rotation,
                                   rotation_mode=rotation_mode, **kwargs)
        return self.tiplabel

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
        self.transProjection = self._gen_transProjection()

        # 2) The above has an output range that is not in the unit rectangle, so
        # scale and translate it.
        self.transAffine = (self._gen_transAffinePart1()
                            + Affine2D().scale(self.height*SQRT2)
                            + Affine2D().translate(0.5, self.height + self.elevation))

        # 3) This is the transformation from axes space to display space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Put these 3 transforms together -- from data all the way to display
        # coordinates.  Using the '+' operator, these transforms are applied
        # "in order".  The transforms are automatically simplified, if possible,
        # by the underlying transformation framework.
        self.transData = self.transProjection + self.transAffine + self.transAxes

        # X-axis gridlines and ticklabels.
        self._xaxis_transform = self.transData
        angle = np.radians(self.angle)
        self._xaxis_text1_transform = (self.transData
                                       + Affine2D().translate(4*np.cos(angle), # 4-pixel offset
                                                              4*np.sin(angle)))

        self._xaxis_text2_transform = IdentityTransform() # Required but not used

        # Y-axis gridlines and ticklabels.
        self._yaxis_transform = self.transData
        self._yaxis_text1_transform = IdentityTransform() # Required but not used
        self._yaxis_text2_transform = IdentityTransform() # Required but not used

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
        vertices = np.array([[0.5, self.elevation + self.height], # Point c=1 (upper center)
                             [0.5 - self.height/SQRT3,  self.elevation], # Point a=1 (lower left)
                             [0.5 + self.height/SQRT3, self.elevation], # Point b=1 (lower right)
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
        self.yaxis = maxis.YAxis(self)
        # Don't register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until xaxis.cla() works.
        #self.spines['ternary'].register_axis(self.xaxis)
        #self.spines['ternary'].register_axis(self.yaxis)
        self._update_transScale()

    def __init__(self, *args, **kwargs):
        self.angle = self.get_angle()

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
        non-separable part of mapping *b* (lower right component) and *c* (upper
        center component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, bc):
            b = bc[:, 0:1]
            c  = bc[:, 1:2]
            x = b
            y = c + b - 1
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
            b = x
            c = y + b - 1
            return np.concatenate((b, c), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryBCAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    def _gen_transProjection(self):
        """Return the projection transformation.

        This is a method so that it can return an affine transformation (the
        identity transformation for the *a*, *b* axes) or a non-affine
        transformation (for the other two axes).
        """
        return self.TernaryTransform()

    def _gen_transAffinePart1(self):
        """This is the part of the affine transformation that is unique to the
        *b*, *c* ternary axes.
        """
        return Affine2D().rotate_deg(-45) + Affine2D().scale(1/SQRT3, 1)

    def get_angle(self):
        """Return the offset angle [deg] of these axes relative to the a, b axes.
        """
        return 240

class TernaryCAAxes(TernaryABAxes):
    name = 'ternaryca'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *c* (upper center component) and *a*
        (lower left component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, ca):
            c = ca[:, 0:1]
            a  = ca[:, 1:2]
            x = a
            y = c + a - 1

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
            a = x
            b = y + 1 - a
            return np.concatenate((a, b), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return TernaryCAAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    def _gen_transProjection(self):
        """Return the projection transformation.

        This is a method so that it can return an affine transformation (the
        identity transformation for the *a*, *b* axes) or a non-affine
        transformation (for the other two axes).
        """
        return self.TernaryTransform()

    def _gen_transAffinePart1(self):
        """This is the part of the affine transformation that is unique to the
        *c*, *a* ternary axes.
        """
        return Affine2D().rotate_deg(135) + Affine2D().scale(1/SQRT3, -1)

    def get_angle(self):
        """Return the offset angle [deg] of these axes relative to the c, a axes.
        """
        return 0
