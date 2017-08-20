"""
matplotlib.lines.Line2D refactored in traitlets
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings
import numpy as np


# from .artist import Artist
from matplotlib import artist, colors as mcolors, docstring, rcParams
from .artist import Artist, allow_rasterization
import matplotlib.artist as b_artist
print('matplotlib.artist b_artist: ', b_artist)


# import matplotlib._traits.artist as artist
# from matplotlib._traits.artist import Artist, allow_rasterization
from .cbook import (
    iterable, is_numlike, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, TransformedPath, IdentityTransform
# Imported here for backward compatibility, even though they don't
# really belong.
from numpy import ma
from . import _path
from matplotlib.markers import (
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)

#for monkey patching into base lines
import matplotlib.lines.Line2D as b_Line2D


class Line2D(HasTraits, b_artist.Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    """

    # remain the same throughout
    lineStyles = _lineStyles = {  # hidden names deprecated
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    #remain the same
    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    #remain the same
    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    # drawStyles should now be deprecated.
    drawStyles = {}
    drawStyles.update(_drawStyles_l)
    drawStyles.update(_drawStyles_s)
    # Need a list ordered with long names first:
    drawStyleKeys = list(_drawStyles_l) + list(_drawStyles_s)

    # Referenced here to maintain API.  These are defined in
    # MarkerStyle
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    zorder = 2
    validCap = ('butt', 'round', 'projecting')
    validJoin = ('miter', 'round', 'bevel')

    xdata=Instance('numpy.array', allow_none=True,default_value=True) # not sure about this line
    ydata=Instance('numpy.array', allow_none=True,default_value=True) # not sure about this line
    linewidth=Float(allow_none=True, default_value=None)
    # linestyle=
    # color=
    marker=Instance('matplotlib.markers',allow_none=True, default_value=None)
    markersize=Float(allow_none=True,default_value=True)
    # markeredgewidth=
    # fillstyle=
    antialiased=Bool(default_value=False)
    # dash_capstyle=
    # solid_capstyle=
    # dash_joinstyle=
    # solid_joinstyle=
    pickradius=Int(allow_none=True, default_value=5)
    # drawstyle=
    # markevery=


    # not sure how much this will have to be refactored
    def __str__(self):
        if self._label != "":
            return "Line2D(%s)" % (self._label)
        elif self._x is None:
            return "Line2D()"
        elif len(self._x) > 3:
            return "Line2D((%g,%g),(%g,%g),...,(%g,%g))"\
                % (self._x[0], self._y[0], self._x[0],
                   self._y[0], self._x[-1], self._y[-1])
        else:
            return "Line2D(%s)"\
                % (",".join(["(%g,%g)" % (x, y) for x, y
                             in zip(self._x, self._y)]))

    #this will have to be edited according to the traits
    def __init__(self, xdata, ydata,
                 linewidth=None,  # all Nones default to rc
                 linestyle=None,
                 color=None,
                 marker=None,
                 markersize=None,
                 markeredgewidth=None,
                 markeredgecolor=None,
                 markerfacecolor=None,
                 markerfacecoloralt='none',
                 fillstyle=None,
                 antialiased=None,
                 dash_capstyle=None,
                 solid_capstyle=None,
                 dash_joinstyle=None,
                 solid_joinstyle=None,
                 pickradius=5,
                 drawstyle=None,
                 markevery=None,
                 **kwargs
                 ):



        """
        Create a :class:`~matplotlib.lines.Line2D` instance with *x*
        and *y* data in sequences *xdata*, *ydata*.

        The kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        See :meth:`set_linestyle` for a decription of the line styles,
        :meth:`set_marker` for a description of the markers, and
        :meth:`set_drawstyle` for a description of the draw styles.
        """

        #initialize Artist in Line2D
        Artist.__init__(self)

        #convert sequences to numpy arrays
        if not iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        if linewidth is None:
            linewidth = rcParams['lines.linewidth']

        if linestyle is None:
            linestyle = rcParams['lines.linestyle']
        if marker is None:
            marker = rcParams['lines.marker']
        if color is None:
            color = rcParams['lines.color']

        if markersize is None:
            markersize = rcParams['lines.markersize']
        if antialiased is None:
            antialiased = rcParams['lines.antialiased']
        if dash_capstyle is None:
            dash_capstyle = rcParams['lines.dash_capstyle']
        if dash_joinstyle is None:
            dash_joinstyle = rcParams['lines.dash_joinstyle']
        if solid_capstyle is None:
            solid_capstyle = rcParams['lines.solid_capstyle']
        if solid_joinstyle is None:
            solid_joinstyle = rcParams['lines.solid_joinstyle']

        if isinstance(linestyle, six.string_types):
            ds, ls = self._split_drawstyle_linestyle(linestyle)
            if ds is not None and drawstyle is not None and ds != drawstyle:
                raise ValueError("Inconsistent drawstyle ({0!r}) and "
                                 "linestyle ({1!r})".format(drawstyle,
                                                            linestyle)
                                 )
            linestyle = ls

            if ds is not None:
                drawstyle = ds

        if drawstyle is None:
            drawstyle = 'default'

        self._dashcapstyle = None
        self._dashjoinstyle = None
        self._solidjoinstyle = None
        self._solidcapstyle = None
        self.set_dash_capstyle(dash_capstyle)
        self.set_dash_joinstyle(dash_joinstyle)
        self.set_solid_capstyle(solid_capstyle)
        self.set_solid_joinstyle(solid_joinstyle)

        self._linestyles = None
        self._drawstyle = None
        self._linewidth = linewidth

        # scaled dash + offset
        self._dashSeq = None
        self._dashOffset = 0
        # unscaled dash + offset
        # this is needed scaling the dash pattern by linewidth
        self._us_dashSeq = None
        self._us_dashOffset = 0

        self.set_linestyle(linestyle)
        self.set_drawstyle(drawstyle)
        self.set_linewidth(linewidth)

        self._color = None
        self.set_color(color)
        self._marker = MarkerStyle(marker, fillstyle)

        self._markevery = None
        self._markersize = None
        self._antialiased = None

        self.set_markevery(markevery)
        self.set_antialiased(antialiased)
        self.set_markersize(markersize)

        self._markeredgecolor = None
        self._markeredgewidth = None
        self._markerfacecolor = None
        self._markerfacecoloralt = None

        self.set_markerfacecolor(markerfacecolor)
        self.set_markerfacecoloralt(markerfacecoloralt)
        self.set_markeredgecolor(markeredgecolor)
        self.set_markeredgewidth(markeredgewidth)

        self.verticalOffset = None

        # update kwargs before updating data to give the caller a
        # chance to init axes (and hence unit support)
        self.update(kwargs)
        self.pickradius = pickradius
        self.ind_offset = 0
        if is_numlike(self._picker):
            self.pickradius = self._picker

        self._xorig = np.asarray([])
        self._yorig = np.asarray([])
        self._invalidx = True
        self._invalidy = True
        self._x = None
        self._y = None
        self._xy = None
        self._path = None
        self._transformed_path = None
        self._subslice = False
        self._x_filled = None  # used in subslicing; only x is needed

        self.set_data(xdata, ydata)
"""
________________________________________________________________________________
END OF INIT FUNCTION
"""

    #xdata default
    @default("xdata")
    def _xdata_default(self):
        print("xdata: generating default value")
        return False
    #xdata validate
    @validate("xdata")
    def _xdata_validate(self, proposal):
        print("xdata: cross validating %r" % proposal.value)
        return proposal.value
    #xdata observer
    @observe("xdata", type="change")
    def _xdata_observe(self, change):
        print("xdata: observed a change from %r to %r" % (change.old, change.new))

#ydata default
    @default("ydata")
    def _ydata_default(self):
        print("ydata: generating default value")
        return False
    #ydata validate
    @validate("ydata")
    def _ydata_validate(self, proposal):
        print("ydata: cross validating %r" % proposal.value)
        return proposal.value
    #ydata observer
    @observe("ydata", type="change")
    def _ydata_observe(self, change):
        print("ydata: observed a change from %r to %r" % (change.old, change.new))

    #linewidth default
    @default("linewidth")
    def _linewidth_default(self):
        print("linewidth: generating default value")
        return False
    #linewidth validate
    @validate("linewidth")
    def _linewidth_validate(self, proposal):
        print("linewidth: cross validating %r" % proposal.value)
        return proposal.value
    #linewidth observer
    @observe("linewidth", type="change")
    def _linewidth_observe(self, change):
        print("linewidth: observed a change from %r to %r" % (change.old, change.new))

    #linestyle default
    @default("linestyle")
    def _linestyle_default(self):
        print("linestyle: generating default value")
        return False
    #linestyle validate
    @validate("linestyle")
    def _linestyle_validate(self, proposal):
        print("linestyle: cross validating %r" % proposal.value)
        return proposal.value
    #linestyle observer
    @observe("linestyle", type="change")
    def _linestyle_observe(self, change):
        print("linestyle: observed a change from %r to %r" % (change.old, change.new))

    #color default
    @default("color")
    def _color_default(self):
        print("color: generating default value")
        return False
    #color validate
    @validate("color")
    def _color_validate(self, proposal):
        print("color: cross validating %r" % proposal.value)
        return proposal.value
    #color observer
    @observe("color", type="change")
    def _color_observe(self, change):
        print("color: observed a change from %r to %r" % (change.old, change.new))

    #marker default
    @default("marker")
    def _marker_default(self):
        print("marker : generating default value")
        return False
    #marker validate
    @validate("marker")
    def _marker_validate(self, proposal):
        print("marker: cross validating %r" % proposal.value)
        return proposal.value
    #marker observer
    @observe("marker", type="change")
    def _marker_observe(self, change):
        print("marker: observed a change from %r to %r" % (change.old, change.new))

    #markersize default
    @default("markersize")
    def _markersize_default(self):
        print("markersize : generating default value")
        return False
    #markersize validate
    @validate("markersize")
    def _markersize_validate(self, proposal):
        print("markersize: cross validating %r" % proposal.value)
        return proposal.value
    #markersize observer
    @observe("markersize", type="change")
    def _markersize_observe(self, change):
        print("markersize: observed a change from %r to %r" % (change.old, change.new))

    #markeredgewidth default
    @default("markeredgewidth")
    def _markeredgewidth_default(self):
        print("markeredgewidth : generating default value")
        return False
    #markeredgewidth validate
    @validate("markeredgewidth")
    def _markeredgewidth_validate(self, proposal):
        print("markeredgewidth: cross validating %r" % proposal.value)
        return proposal.value
    #markeredgewidth observer
    @observe("markeredgewidth", type="change")
    def _markeredgewidth_observe(self, change):
        print("markeredgewidth: observed a change from %r to %r" % (change.old, change.new))

    #fillstyle default
    @default("fillstyle")
    def _fillstyle_default(self):
        print("fillstyle : generating default value")
        return False
    #fillstyle validate
    @validate("fillstyle")
    def _fillstyle_validate(self, proposal):
        print("fillstyle: cross validating %r" % proposal.value)
        return proposal.value
    #fillstyle observer
    @observe("fillstyle", type="change")
    def _fillstyle_observe(self, change):
        print("fillstyle: observed a change from %r to %r" % (change.old, change.new))

    #antialiased default
    @default("antialiased")
    def _antialiased_default(self):
        print("antialiased : generating default value")
        return False
    #antialiased validate
    @validate("antialiased")
    def _antialiased_validate(self, proposal):
        print("antialiased: cross validating %r" % proposal.value)
        return proposal.value
    #antialiased observer
    @observe("antialiased", type="change")
    def _antialiased_observe(self, change):
        print("antialiased: observed a change from %r to %r" % (change.old, change.new))

    #dash_capstyle default
    @default("dash_capstyle")
    def _dash_capstyle_default(self):
        print("dash_capstyle : generating default value")
        return False
    #dash_capstyle validate
    @validate("dash_capstyle")
    def _dash_capstyle_validate(self, proposal):
        print("dash_capstyle: cross validating %r" % proposal.value)
        return proposal.value
    #dash_capstyle observer
    @observe("dash_capstyle", type="change")
    def _dash_capstyle_observe(self, change):
        print("dash_capstyle: observed a change from %r to %r" % (change.old, change.new))

    #solid_capstyle default
    @default("solid_capstyle")
    def _solid_capstyle_default(self):
        print("solid_capstyle : generating default value")
        return False
    #solid_capstyle validate
    @validate("solid_capstyle")
    def _solid_capstyle_validate(self, proposal):
        print("solid_capstyle: cross validating %r" % proposal.value)
        return proposal.value
    #solid_capstyle observer
    @observe("solid_capstyle", type="change")
    def _solid_capstyle_observe(self, change):
        print("solid_capstyle: observed a change from %r to %r" % (change.old, change.new))

    #dash_joinstyle default
    @default("dash_joinstyle")
    def _dash_joinstyle_default(self):
        print("dash_joinstyle : generating default value")
        return False
    #dash_joinstyle validate
    @validate("dash_joinstyle")
    def _dash_joinstyle_validate(self, proposal):
        print("dash_joinstyle: cross validating %r" % proposal.value)
        return proposal.value
    # observer
    @observe("dash_joinstyle", type="change")
    def __dash_joinstyleobserve(self, change):
        print(": observed a change from %r to %r" % (change.old, change.new))

    #solid_joinstyle default
    @default("solid_joinstyle")
    def _solid_joinstyle_default(self):
        print("solid_joinstyle : generating default value")
        return False
    #solid_joinstyle validate
    @validate("solid_joinstyle")
    def _solid_joinstyle_validate(self, proposal):
        print("solid_joinstyle: cross validating %r" % proposal.value)
        return proposal.value
    #solid_joinstyle observer
    @observe("solid_joinstyle", type="change")
    def _solid_joinstyle_observe(self, change):
        print("solid_joinstyle: observed a change from %r to %r" % (change.old, change.new))

    #pickradius default
    @default("pickradius")
    def _pickradius_default(self):
        print("pickradius : generating default value")
        return False
    #pickradius validate
    @validate("pickradius")
    def _pickradius_validate(self, proposal):
        print("pickradius: cross validating %r" % proposal.value)
        return proposal.value
    #pickradius observer
    @observe("pickradius", type="change")
    def _pickradius_observe(self, change):
        print("pickradius: observed a change from %r to %r" % (change.old, change.new))

    #drawstyle default
    @default("drawstyle")
    def _drawstyle_default(self):
        print("drawstyle : generating default value")
        return False
    #drawstyle validate
    @validate("drawstyle")
    def _drawstyle_validate(self, proposal):
        print("drawstyle: cross validating %r" % proposal.value)
        return proposal.value
    #drawstyle observer
    @observe("drawstyle", type="change")
    def _drawstyle_observe(self, change):
        print("drawstyle: observed a change from %r to %r" % (change.old, change.new))

    #markevery default
    @default("markevery")
    def _markevery_default(self):
        print("markevery : generating default value")
        return False
    #markevery validate
    @validate("markevery")
    def _markevery_validate(self, proposal):
        print("markevery: cross validating %r" % proposal.value)
        return proposal.value
    #markevery observer
    @observe("markevery", type="change")
    def _markevery_observe(self, change):
        print("markevery: observed a change from %r to %r" % (change.old, change.new))



#for monkey patching
b_Line2D = Line2D()
