#!/usr/bin/env python
"""matplotlib projections for ternary axes
"""
__author__ = "Kevin L. Davies"
__version__ = "2011/10/12"
__license__ = "BSD"
# The features and concepts were based on triangleplot.py by C. P. H. Lewis,
# 2008-2009, available at:
#     http://nature.berkeley.edu/~chlewis/Sourcecode.html
# The projection class and methods were adapted from
# custom_projection_example.py, available at:
#     http://matplotlib.sourceforge.net/examples/api/custom_projection_example.html

# **To do:
#   1. Clean up the procedure for setting the sum of a, b, and c (the total
#      should be automatically the same for all 3 axes).
#   2. Clean up the code and document it (in ReST format; give a summary of
#      features/changes relative to previous work).
#   3. Post it to github for review, modify, and submit a pull request.

# **Nice to have:
#   1. Add automatic offsetting/padding of tiplabels.
#   2. Allow zooming/setting of axes limits (upon updating one axis, apply the
#      same delta to other axes, but retain the other axes' minimum values).
#   3. Add the option for clockwise-increasing axes (through invert_xaxis?)

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
#         plotfile, table, tick_params, ticklabel_format, xlim, xticks
#     Not applicable (y-axis isn't shown):
#         ylabel, ylim, yticks
#     Not applicable and disabled:
#         zgrids, thetagrids

# Colormap methods (should be also be compatible, but **should be tested):
#     autumn, bone, cool, copper, flag, gray, hot, hsv, jet, pink, prism,
#     spring, summer, winter, and spectral.

import numpy as np

import matplotlib
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import matplotlib.text as mtext
import matplotlib.font_manager as font_manager
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.cbook as cbook
import matplotlib.markers as mmarkers
import matplotlib.ticker as mticker

from matplotlib import rcParams
#from matplotlib.cbook import _string_to_bool
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform, \
                                  IdentityTransform, Bbox
from matplotlib.markers import MarkerStyle

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)


class Line2D(mlines.Line2D):
    """Customizations to the Line2D class
    """
    def update_from(self, other):
        """Copy the marker properties too.
        """
        mlines.Line2D.update_from(self, other)
        self._marker = other._marker


class XTick(maxis.XTick):
    """Customizations to the x-tick methods for ternary axes
    """
    def _get_text1(self):
        """Get the default Text instance.
        """
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

    def _get_tick1line(self):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D(xdata=(0,), ydata=(0,),
                   color=self._color,
                   linestyle = 'None',
                   marker = self._tickmarkers[0],
                   markersize=self._size,
                   markeredgewidth=self._width,
                   zorder=self._zorder,
                   )
        l._marker._transform += Affine2D().rotate_deg(self.axes.angle + 90)
        l.set_transform(self.axes.get_xaxis_transform(which='tick1'))
        self._set_artist_props(l)
        return l


class XAxis(maxis.XAxis):
    """Customizations to the x-axis methods for ternary axes

    The x-axis is used for component A and is on the right side.
    """
    def set_label_position(self, position):
        raise NotImplementedError("Label positions are fixed in the ternary "
                                  "axes.")
        return

    def _get_label(self):
        # x and y are both in display coordinates.  This is non-conventional.
        # **Is this ok?
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
        if len(bboxes) and self._autolabelpos:
            # Find the center of the x axis.
            x0, y0 = self.axes.transData.transform_point((0, 0))
            x1, y1 = self.axes.transData.transform_point((self.axes.total, 0))
            x = (x0 + x1)/2.0
            y = (y0 + y1)/2.0

            # Account for the the maximum possible length of tick labels.
            max_width = 0
            max_height = 0
            for bbox in bboxes:
                max_width = max(max_width, bbox.width)
                max_height = max(max_height, bbox.height)
            space = np.sqrt(max_width**2 + max_height**2)
            space += self.labelpad*self.figure.dpi/72.0

            # Apply the new position.
            angle = np.radians(self.axes.angle) # Offset angle in radians
            self.label.set_position((x + np.cos(angle)*space,
                                     y + np.sin(angle)*space))

    def _get_tick(self, major):
        """Return the default tick instance (copied here to use the XTick class
        from this file).
        """
        if major:
            tick_kw = self._major_tick_kw
        else:
            tick_kw = self._minor_tick_kw
        return XTick(self.axes, 0, '', major=major, **tick_kw)

    def set_ticks(self, ticks, minor=False):
        """
        Set the locations of the tick marks from sequence ticks

        ACCEPTS: sequence of floats
        """
        ### XXX if the user changes units, the information will be lost here
        ticks = self.convert_units(ticks)
        if len(ticks) > 1:
            xleft, xright = self.get_view_interval()
            #if xright > xleft:
            #    self.set_view_interval(min(ticks), max(ticks))
            #else:
            #    self.set_view_interval(max(ticks), min(ticks))
        if minor:
            self.set_minor_locator(mticker.FixedLocator(ticks))
            return self.get_minor_ticks(len(ticks))
        else:
            self.set_major_locator( mticker.FixedLocator(ticks) )
            return self.get_major_ticks(len(ticks))

class Spine(mspines.Spine):
    """Customizations to matplotlib.spines.Spine"
    """
    @classmethod
    def triangular_spine(cls, axes, spine_type, **kwargs):
        """(staticmethod) Returns a linear :class:`Spine` for the ternary axes.

        Based on linear_spine()
        """
        # The 'bottom' and 'left' are swapped here so that they appear
        # correctly.  The 'bottom' spine is associated with the x-axis, which is
        # on the left in the ternary plot.  The 'left' spine is associated with
        # the y-axis, which is the bottom axis in the ternary plot.
        if spine_type == 'bottom':
            path = Path([(0.0, 0.0), (1.0, 0.0)]) # Actually the left axis.
        elif spine_type == 'left':
            path = Path([(1.0, 0.0), (0.0, 1.0)]) # Actually the bottom axis.
        elif spine_type == 'right':
            path = Path([(0.0, 1.0), (0.0, 0.0)])
        else:
            raise ValueError("Unable to make path for spine " + spine_type)
        return cls(axes, spine_type, path, **kwargs)


class TernaryABAxes(Axes):
    """A matplotlib projection for ternary axes

    Ternary plots are useful for plotting sets of three variables that each sum
    to the same value.  For an introduction, see
    http://en.wikipedia.org/wiki/Ternary_plot.

    In general, ternary plots may be rendered with clockwise- or
    counterclockwise-increasing axes.  Here, the ternary projections use the
    counterclockwise convention.  Specifically, the ternaryab projection maps
    *a* and *b* in ternary space to *x* and *y* in Cartesian space.
    """
    name = 'ternaryab'

    def _gen_transTernary(self):
        """Return the ternary transformation.

        This is implemented as a method so that it can return an affine
        transformation (the identity transformation for the *a*, *b* axes) or a
        non-affine transformation (for the other two axes).
        """
        return IdentityTransform()

    def _gen_transAffinePart1(self):
        """This is the part of the affine transformation that is unique to the
        *a*, *b* ternary axes.
        """
        return Affine2D().rotate_deg(225) + Affine2D().scale(1/SQRT3, 1)

    def get_angle(self):
        """Return the angle [deg] of these axes relative to the *c*, *a* axes.
        """
        return 120

    # Prevent the user from applying nonlinear scales to either of the axes
    # since that would be confusing to the viewer. (The axes should have the
    # same scales, yet all three cannot be nonlinear at once.)
    def set_xscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_xscale(self, *args, **kwargs)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_yscale(self, *args, **kwargs)

    # Disable changes to the data limits.  **Can this be supported?
    def set_xlim(self, *args, **kwargs):
    # ** work in progress...
        #Axes.set_xlim(self, *args, **kwargs)
        #Axes.set_ylim(self, *args, **kwargs)
        #self._set_lim_and_transforms()
        #self.total = self.viewLim.intervalx[1]
        pass
    def set_ylim(self, *args, **kwargs):
    # ** work in progress...
        #Axes.set_xlim(self, *args, **kwargs)
        #Axes.set_ylim(self, *args, **kwargs)
        #self._set_lim_and_transforms()
        pass

    # Disable interactive panning and zooming.  **Can this be supported?
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

    # Customized methods for the ternary axes
    def twinx(self, projection):
        """call signature::

          ax = twinx()

        Create a twin of the current ternary axes with a different projection,
        so that all of the indexing schemes can be used -- (*a*, *b*),
        (*b*, *c*), and (*c*, *a*).
        """
        return self.figure.add_axes(self.get_position(True), sharex=self,
                                    projection=projection, frameon=False)

    def legend(self, *args, **kwargs):
        """call signature::

          legend(*args, **kwargs)

        Place a legend on the current axes at location *loc*.  Labels are a
        sequence of strings and *loc* can be a string or an integer specifying
        the legend location.

        To make a legend with existing lines::

          legend()

        :meth:`legend` by itself will try and build a legend using the label
        property of the lines/patches/collections.  You can set the label of
        a line by doing::

          plot(x, y, label='my data')

        or::

          line.set_label('my data').

        If label is set to '_nolegend_', the item will not be shown in
        legend.

        To automatically generate the legend from labels::

          legend( ('label1', 'label2', 'label3') )

        To make a legend for a list of lines and labels::

          legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

        To make a legend at a given location, using a location argument::

          legend( ('label1', 'label2', 'label3'), loc='upper left')

        or::

          legend( (line1, line2, line3),  ('label1', 'label2', 'label3'), loc=2)

        The location codes are

          ===============   =============
          Location String   Location Code
          ===============   =============
          'best'            0
          'upper right'     1
          'upper left'      2
          'lower left'      3
          'lower right'     4
          'right'           5
          'center left'     6
          'center right'    7
          'lower center'    8
          'upper center'    9
          'center'          10
          ===============   =============


        Users can specify any arbitrary location for the legend using the
        *bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance
        of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.
        For example,

          loc = 'upper right', bbox_to_anchor = (0.5, 0.5)

        will place the legend so that the upper right corner of the legend at
        the center of the axes.

        The legend location can be specified in other coordinate, by using the
        *bbox_transform* keyword.

        The loc itslef can be a 2-tuple giving x,y of the lower-left corner of
        the legend in axes coords (*bbox_to_anchor* is ignored).


        Keyword arguments:

          *prop*: [ None | FontProperties | dict ]
            A :class:`matplotlib.font_manager.FontProperties`
            instance. If *prop* is a dictionary, a new instance will be
            created with *prop*. If *None*, use rc settings.

          *numpoints*: integer
            The number of points in the legend for line

          *scatterpoints*: integer
            The number of points in the legend for scatter plot

          *scatteroffsets*: list of floats
            a list of yoffsets for scatter symbols in legend

          *markerscale*: [ None | scalar ]
            The relative size of legend markers vs. original. If *None*, use rc
            settings.

          *frameon*: [ True | False ]
            if True, draw a frame around the legend.
            The default is set by the rcParam 'legend.frameon'

          *fancybox*: [ None | False | True ]
            if True, draw a frame with a round fancybox.  If None, use rc

          *shadow*: [ None | False | True ]
            If *True*, draw a shadow behind legend. If *None*, use rc settings.

          *ncol* : integer
            number of columns. default is 1

          *mode* : [ "expand" | None ]
            if mode is "expand", the legend will be horizontally expanded
            to fill the axes area (or *bbox_to_anchor*)

          *bbox_to_anchor* : an instance of BboxBase or a tuple of 2 or 4 floats
            the bbox that the legend will be anchored.

          *bbox_transform* : [ an instance of Transform | None ]
            the transform for the bbox. transAxes if None.

          *title* : string
            the legend title

        Padding and spacing between various elements use following
        keywords parameters. These values are measure in font-size
        units. E.g., a fontsize of 10 points and a handlelength=5
        implies a handlelength of 50 points.  Values from rcParams
        will be used if None.

        ================   ==================================================================
        Keyword            Description
        ================   ==================================================================
        borderpad          the fractional whitespace inside the legend border
        labelspacing       the vertical space between the legend entries
        handlelength       the length of the legend handles
        handletextpad      the pad between the legend handle and text
        borderaxespad      the pad between the axes and legend border
        columnspacing      the spacing between columns
        ================   ==================================================================

        .. Note:: Not all kinds of artist are supported by the legend command.
                  See LINK (FIXME) for details.


        **Example:**

        .. plot:: mpl_examples/api/legend_demo.py

        Also see :ref:`plotting-guide-legend`.

        """
        # The legend needs to be positioned outside the bounding box of the plot
        # area.  The normal specifications (e.g., legend(loc='upper right')) do
        # not do this.
        loc=kwargs.pop('loc', 'upper left')
        borderaxespad=kwargs.pop('borderaxespad', 0)
        # The left edge of the legend is horizontally aligned with the right
        # vertex of the plot.  The top edge of the legend is vertically aligned
        # with the top vertex of the plot.
        bbox_to_anchor=kwargs.pop('bbox_to_anchor', (0.5 + self.height/SQRT3,
                                                     self.elevation + self.height))
        Axes.legend(self, loc=loc, borderaxespad=borderaxespad,
                    bbox_to_anchor=bbox_to_anchor, *args, **kwargs)

    def _set_total(self, total):
        """Set the total of a, b, and c.
        """
        self.total = total
        self._set_lim_and_transforms()
        # This is a hack.  **Clean it up.
        self.xaxis.set_ticklabels(['%g'%val for val in np.linspace(0, self.total, 5)])
        #self.set_xticks(np.linspace(0, self.total, 5))
        #self.set_xlim(0.0, self.total)

    def cla(self):
        """Provide reasonable defaults for the axes.
        """
        # Call the base class.
        Axes.cla(self)
        self.grid(True)

        # Only the x-axis is shown, but there are 3 axes once all of the
        # projections are included.
        self.yaxis.set_visible(False)

        # Adjust the number of ticks shown.
        #self.set_xticks(np.linspace(0, self.viewLim.x1, 5))
        self.set_xticks(np.linspace(0, self.total, 5))

        # Turn off minor ticking altogether.
        self.xaxis.set_minor_locator(NullLocator())

        # Place the title a little higher than normal.
        self.title.set_y(1.02)

        # Modify the padding between the tick labels and the axis labels.
        self.xaxis.labelpad = 10 # In display units

        # Spacing from the vertices (tips) to the tip labels (in data coords, as
        # a fraction of self.total)
        self.tipoffset = 0.14

    def set_tiplabel(self, tiplabel, ha='center', va='center',
                   rotation_mode='anchor', **kwargs):
        """Add a tip label for the first component.
        """
        tipoffset = kwargs.pop('tipoffset', self.tipoffset)
        rotation = kwargs.pop('rotation', self.angle)
        self.tiplabel = self.text((1 + tipoffset)*self.total,
                                  -tipoffset*self.total/2.0, tiplabel,
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
        #print self.xaxis.view_interval
        self.transProjection = (Affine2D().translate(-self.viewLim.x0, -self.viewLim.y0)
                                + Affine2D().scale(1.0/self.total)
                                + self._gen_transTernary())

        # 2) The above has an output range that is not in the unit rectangle, so
        # scale and translate it.
        self.transAffine = (self._gen_transAffinePart1()
                            + Affine2D().scale(self.height*SQRT2)
                            + Affine2D().translate(0.5, self.height + self.elevation))

        # 3) This is the transformation from axes space to display space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Put these 3 transforms together -- from data all the way to display
        # coordinates.  The parentheses are important for efficiency -- they
        # group the last two (which are usually affines) separately from the
        # first (which can be non-affine).
        self.transData = self.transProjection + (self.transAffine
                                                 + self.transAxes)

        # X-axis gridlines and ticklabels.
        self._xaxis_transform = self.transData

 #       self._xaxis_transform = Affine2D().rotate_deg(90) + self.transData
        #self._xaxis_transform = IdentityTransform()
        angle = np.radians(self.angle)
        self._xaxis_text1_transform = (self.transData
                                       + Affine2D().translate(4*np.cos(angle), # 4-pixel offset
                                                              4*np.sin(angle)))
        self._xaxis_text2_transform = IdentityTransform() # Required but not used

        # Y-axis gridlines and ticklabels.
#        self._yaxis_transform = Affine2D().rotate_deg(-180) + self.transData
        self._yaxis_transform = IdentityTransform() # Required but not used
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
#        return {'ternary1':Spine.triangular_spine(self, 'bottom'),
#                'ternary2':Spine.triangular_spine(self, 'left'),
#                'ternary3':Spine.triangular_spine(self, 'right')}
        return {'ternary1':Spine.triangular_spine(self, 'left'),
                'ternary2':Spine.triangular_spine(self, 'bottom'),
                'ternary3':Spine.triangular_spine(self, 'right')}

    def _init_axis(self):
        self.xaxis = XAxis(self)
        self.yaxis = maxis.YAxis(self)

        # Don't draw ticks on the secondary x axis.
        self.xaxis._major_tick_kw['tick2On'] = False

        # Don't register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until xaxis.cla() works.
        #self.spines['ternaryab'].register_axis(self.xaxis)
        #self.spines['ternaryab'].register_axis(self.yaxis)
        self._update_transScale()

    def __init__(self, *args, **kwargs):
        # Sum of a, b, and c
        self.total = 1.0

        # Offset angle [deg] of these axes relative to the c, a axes.
        self.angle = self.get_angle()

        # Height of the ternary outline (in axis units)
        self.height = 0.8

        # Vertical distance between the y axis and the base of the ternary plot
        # (in axis units)
        self.elevation = 0.05

        Axes.__init__(self, *args, **kwargs)
        self.cla()
        self.set_aspect(aspect='equal', adjustable='box-forced', anchor='C') # C is for center.


class TernaryBCAxes(TernaryABAxes):
    """A matplotlib projection for ternary axes

    Ternary plots are useful for plotting sets of three variables that each sum
    to the same value.  For an introduction, see
    http://en.wikipedia.org/wiki/Ternary_plot.

    In general, ternary plots may be rendered with clockwise- or
    counterclockwise-increasing axes.  Here, the ternary projections use the
    counterclockwise convention.  Specifically, the ternarybc projection maps
    *b* and *c* in ternary space to *x* and *y* in Cartesian space.
    """
    name = 'ternarybc'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *b* (lower right component) and *c* (upper
        center component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform_non_affine(self, bc):
            b = bc[:, 0:1]
            c  = bc[:, 1:2]
            x = b
            y = c + b - 1
            return np.concatenate((x, y), 1)
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        if matplotlib.__version__ < '1.2':
            # Note: For compatibility with matplotlib v1.1 and older, you'll
            # need to explicitly implement a ``transform`` method as well.
            # Otherwise a ``NotImplementedError`` will be raised. This isn't
            # necessary for v1.2 and newer, however.
            transform = transform_non_affine

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

        def transform_non_affine(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            b = x
            c = y + b - 1
            return np.concatenate((b, c), 1)
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        # As before, we need to implement the "transform" method for
        # compatibility with matplotlib v1.1 and older.
        if matplotlib.__version__ < '1.2':
            transform = transform_non_affine


        def inverted(self):
            return TernaryBCAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    def _gen_transTernary(self):
        """Return the ternary transformation.

        This is implemented as a method so that it can return an affine
        transformation (the identity transformation for the *a*, *b* axes) or a
        non-affine transformation (for the other two axes).
        """
        return self.TernaryTransform()

    def _gen_transAffinePart1(self):
        """Return the offset angle [deg] of these axes relative to the *c*, *a*
        axes.
        """
        return Affine2D().rotate_deg(-45) + Affine2D().scale(1/SQRT3, 1)

    def get_angle(self):
        """Return the angle [deg] of these axes relative to the  *c*, *a* axes.
        """
        return 240

class TernaryCAAxes(TernaryABAxes):
    """A matplotlib projection for ternary axes

    Ternary plots are useful for plotting sets of three variables that each sum
    to the same value.  For an introduction, see
    http://en.wikipedia.org/wiki/Ternary_plot.

    In general, ternary plots may be rendered with clockwise- or
    counterclockwise-increasing axes.  Here, the ternary projections use the
    counterclockwise convention.  Specifically, the ternaryca projection maps
    *c* and *a* in ternary space to *x* and *y* in Cartesian space.
    """
    name = 'ternaryca'

    class TernaryTransform(Transform):
        """This is the core of the ternary transform; it performs the
        non-separable part of mapping *c* (upper center component) and *a*
        (lower left component) into Cartesian coordinate space *x* and *y*.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform_non_affine(self, ca):
            c = ca[:, 0:1]
            a  = ca[:, 1:2]
            x = a
            y = c + a - 1

            return np.concatenate((x, y), 1)
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        if matplotlib.__version__ < '1.2':
            # Note: For compatibility with matplotlib v1.1 and older, you'll
            # need to explicitly implement a ``transform`` method as well.
            # Otherwise a ``NotImplementedError`` will be raised. This isn't
            # necessary for v1.2 and newer, however.
            transform = transform_non_affine

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

        def transform_non_affine(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            a = x
            b = y + 1 - a
            return np.concatenate((a, b), 1)
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        # As before, we need to implement the "transform" method for
        # compatibility with matplotlib v1.1 and older.
        if matplotlib.__version__ < '1.2':
            transform = transform_non_affine

        def inverted(self):
            return TernaryCAAxes.TernaryTransform()
        inverted.__doc__ = Transform.inverted.__doc__

    def _gen_transTernary(self):
        """Return the ternary transformation.

        This is implemented as a method so that it can return an affine
        transformation (the identity transformation for the *a*, *b* axes) or a
        non-affine transformation (for the other two axes).
        """
        return self.TernaryTransform()

    def _gen_transAffinePart1(self):
        """This is the part of the affine transformation that is unique to the
        *c*, *a* ternary axes.
        """
        return Affine2D().rotate_deg(135) + Affine2D().scale(1/SQRT3, -1)

    def get_angle(self):
        """Return the angle [deg] of these axes relative to the *c*, *a* axes.
        """
        return 0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Now make a simple example using the custom projection.
    plt.subplot(111, projection="ternaryab")
    p = plt.plot([-1, 1, 1], [-1, -1, 1], "o-")
    plt.grid(True)

    plt.show()
    
