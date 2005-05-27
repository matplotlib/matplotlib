"""
Classes for the ticks and x and y axis
"""
from __future__ import division
import sys, math, re, copy
from numerix import arange, array, asarray, ones, zeros, \
     nonzero, take, Float, log10, logical_and, \
     dot, sin, cos, tan, pi, sqrt

from artist import Artist
from cbook import enumerate, silent_list, popall
from lines import Line2D, TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN
from mlab import linspace
from matplotlib import rcParams
from patches import bbox_artist
from ticker import NullFormatter, FixedFormatter, ScalarFormatter, LogFormatter
from ticker import NullLocator, FixedLocator, LinearLocator, LogLocator, AutoLocator

from transforms import Value, blend_xy_sep_transform,\
     translation_transform, bbox_all, identity_transform
from font_manager import FontProperties
from text import Text, _process_text_args
from patches import bbox_artist

import pdb
import sys

# Extracted from Text's method to serve as a function
def get_rotation(rotation):
    'return the text angle as float'
    if rotation in ('horizontal', None):
        angle = 0.
    elif rotation == 'vertical':            
        angle = 90.
    else:
        angle = float(rotation)
    return angle%360

class TextWithDash(Artist):
    """
    This is basically a Text with a dash (drawn with a Line2D)
    before/after it. It is intended to be a drop-in replacement
    for Text (implemented using delegation and get/setattr),
    and should behave identically to Text when dashlength=0.0.

    dashlength is the length of the dash in canvas units.

    dashdirection is one of 0 or 1, where 0 draws the dash
    after the text and 1 before. The default is 0.

    dashrotation specifies the rotation of the dash, and
    should generally stay None. In this case
    self.get_dashrotation() returns self.get_rotation().
    (I.e., the dash takes its rotation from the text's
    rotation). Because the text center is projected onto
    the dash, major deviations in the rotation cause
    what may be considered visually unappealing results.

    dashpad is a padding length to add (or subtract) space
    between the text and the dash.

    NOTE: The alignment of the two objects is based on the
    bbox of the Text, as obtained by get_window_extent().
    This, in turn, appears to depend on the font metrics
    as given by the rendering backend. Hence the quality
    of the "centering" of the label text with respect to
    the dash varies depending on the backend used.

    NOTE2: I'm not sure that I got the get_window_extent()
    right, or whether that's sufficient for providing the
    object bbox.
    """
    __name__ = 'textwithdash'

    def __init__(self, 
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='center',
                 horizontalalignment='center',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 dashlength=0.0,
                 dashdirection=0,
                 dashrotation=None,
                 dashpad=3,
                 xaxis=True,
                 ):
        Artist.__init__(self)
        # The position (x,y) values for _mytext and dashline
        # are bogus as given in the instantiation; they will
        # be set correctly by update_coords() in draw()
        self._mytext = Text(
                      x=x, y=y, text=text,
                      color=color,
                      verticalalignment=verticalalignment,
                      horizontalalignment=horizontalalignment,
                      multialignment=multialignment,
                      fontproperties=fontproperties,
                      rotation=rotation,
                      )
        self.dashline = Line2D(xdata=(x, x),
                               ydata=(y, y),
                               color='k',
                               linestyle='-')
        self._x = x
        self._y = y
        self._dashlength = dashlength
        self._dashdirection = dashdirection
        self._dashrotation = dashrotation
        self._dashpad = dashpad
        
    def __getattr__(self, name):
        """Delegate most things to self._mytext.
        """
        # We need to protect for the __init__ case
        if '_mytext' in self.__dict__:
            return getattr(self._mytext, name)

    def __setattr__(self, name, value):
        """Capture only a few things as necessary
        and send the rest off to self._mytext
        """
        if name in ['_mytext', '_dashlength',
                    '_dashdirection', '_dashrotation',
                    '_x', '_y',
                    '_transform', '_transformSet',
                    'figure',                    ]:
            self.__dict__[name] = value
        else:
            # We need to protect for the init case
            text = self._mytext
            if text != None:
                setattr(self._mytext, name, value)

    def draw(self, renderer):
        if not self.get_visible(): return
        self.update_coords(renderer)
        self._mytext.draw(renderer)
        #bbox_artist(self._mytext, renderer, props={'pad':0}, fill=False)
        if self.get_dashlength() > 0.0:
            self.dashline.draw(renderer)

    def update_coords(self, renderer):
        """Computes the actual x,y coordinates for
        self._mytext based on the input x,y and the
        dashlength. Since the rotation is with respect
        to the actual canvas's coordinates we need to
        map back and forth.
        """
        (x, y) = self.get_position()
        dashlength = self.get_dashlength()

        # Shortcircuit this process if we don't have a dash
        if dashlength == 0.0:
            self._mytext.set_position((x, y))
            return

        dashrotation = self.get_dashrotation()
        dashdirection = self.get_dashdirection()
        dashpad = self.get_dashpad()
        transform = self.get_transform()
        
        angle = get_rotation(dashrotation)
        theta = pi*(angle/180.0+dashdirection-1)
        cos_theta, sin_theta = cos(theta), sin(theta)

        # Compute the dash end points
        # The 'c' prefix is for canvas coordinates
        c1 = array(transform.xy_tup((x, y)))
        c2 = c1+dashlength*array([cos_theta, sin_theta])
        (x2, y2) = transform.inverse_xy_tup(tuple(c2))
        self.dashline.set_data((x, x2), (y, y2))

        # We now need to extend this vector out to
        # the center of the text area.
        # The basic problem here is that we're "rotating"
        # two separate objects but want it to appear as
        # if they're rotated together.
        # This is made non-trivial because of the
        # interaction between text rotation and alignment -
        # text alignment is based on the bbox after rotation.
        # We reset/force both alignments to 'center'
        # so we can do something relatively reasonable.
        # There's probably a better way to do this by
        # embedding all this in the object's transformations,
        # but I don't grok the transformation stuff
        # well enough yet.
        we = self._mytext.get_window_extent(renderer=renderer)
        w, h = we.width(), we.height()
        # Watch for zeros
        if sin_theta == 0.0:
            dx = w
            dy = 0.0
        elif cos_theta == 0.0:
            dx = 0.0
            dy = h
        else:
            tan_theta = sin_theta/cos_theta
            dx = w
            dy = w*tan_theta
            if dy > h or dy < -h:
                dy = h
                dx = h/tan_theta
        cwd = array([dx, dy])/2
        cwd *= 1+dashpad/sqrt(dot(cwd,cwd))
        cw = c2+(dashdirection*2-1)*cwd
        self._mytext.set_position(transform.inverse_xy_tup(tuple(cw)))

        # Now set the window extent
        # I'm not at all sure this is the right way to do this.
        we = self._mytext.get_window_extent(renderer=renderer)
        self._window_extent = we.deepcopy()
        self._window_extent.update(((c1[0], c1[1]),), False)
        
        # Finally, make text align center
        self._mytext.set_horizontalalignment('center')
        self._mytext.set_verticalalignment('center')

    def get_window_extent(self, renderer=None):
        if self.get_dashlength() == 0.0:
            return self._mytext.get_window_extent(renderer=renderer)
        else:
            self.update_coords(renderer)
            return self._window_extent
        
    def get_dashlength(self):
        return self._dashlength

    def set_dashlength(self, dl):
        """
Set the length of the dash following the text

ACCEPTS: float
"""
        self._dashlength = dl

    def get_dashdirection(self):
        return self._dashdirection

    def set_dashdirection(self, dd):
        """
Set the direction of the dash following the text.
1 is before the text and 0 is after. The default
is 0, which is what you'd want for the typical
case of ticks below and on the left of the figure.

ACCEPTS: int
"""
        self._dashdirection = dd

    def get_dashrotation(self):
        if self._dashrotation == None:
            return self.get_rotation()
        else:
            return self._dashrotation

    def set_dashrotation(self, dr):
        """
Set the rotation of the dash following the text

ACCEPTS: float
"""
        self._dashrotation = dr

    def get_dashpad(self):
        return self._dashpad

    def set_dashpad(self, dp):
        """
Set the pad of the dash following the text

ACCEPTS: float
"""
        self._dashpad = dp

    def get_position(self):
        "Return x, y as tuple"
        return self._x, self._y

    def set_position(self, xy):
        """
Set the xy position of the text

ACCEPTS: (x,y)
"""
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):        
        """
Set the x position of the text

ACCEPTS: float
"""
        try: self._x.set(x)
        except AttributeError: self._x = x

    def set_y(self, y):
        """
Set the y position of the text

ACCEPTS: float
"""
        try: self._y.set(y)
        except AttributeError: self._y = y

    def set_transform(self, t):
        """
set the Transformation instance used by this artist

ACCEPTS: a matplotlib.transform transformation instance
"""
        self._transform = t
        self._transformSet = True
        self._mytext.set_transform(t)
        self.dashline.set_transform(t)

    def get_figure(self):
        'return the figure instance'
        return self.figure
    
    def set_figure(self, fig):
        """
Set the figure instance the artist belong to

ACCEPTS: a matplotlib.figure.Figure instance
        """
        self.figure = fig
        self._mytext.set_figure(fig)
        self.dashline.set_figure(fig)


class Tick(Artist):
    """
    Abstract base class for the axis ticks, grid lines and labels
    
    1 refers to the bottom of the plot for xticks and the left for yticks
    2 refers to the top of the plot for xticks and the right for yticks

    Publicly accessible attributes
    
      tick1line  : a Line2D instance
      tick2line  : a Line2D instance
      gridline   : a Line2D instance
      label1     : a Text instance
      label2     : a Text instance
      gridOn     : a boolean which determines whether to draw the tickline
      tick1On    : a boolean which determines whether to draw the 1st tickline
      tick2On    : a boolean which determines whether to draw the 2nd tickline
      label1On   : a boolean which determines whether to draw tick label
      label2On   : a boolean which determines whether to draw tick label
      
      
    """
    def __init__(self, axes, loc, label,
                 size   = None,  # points
                 gridOn = None, # defaults to axes.grid
                 tick1On = True,
                 tick2On = True,
                 label1On = True,
                 label2On = False,
                 major = True,
                 ):
        """
        bbox is the Bound2D bounding box in display coords of the Axes
        loc is the tick location in data coords
        size is the tick size in relative, axes coords
        """
        Artist.__init__(self)

        if gridOn is None: gridOn = rcParams['axes.grid']

        self.set_figure(axes.figure)
        self.axes = axes

        if size is None:
            if major:
                size = rcParams['tick.major.size']
                pad = rcParams['tick.major.pad']
            else:
                size = rcParams['tick.minor.size']
                pad = rcParams['tick.minor.pad']

        self._loc = loc
        self._size = size
        self._pad = Value(pad)
        
        self._padPixels = self.figure.dpi*self._pad*Value(1/72.0) 


        self.tick1line = self._get_tick1line(loc)
        self.tick2line = self._get_tick2line(loc)
        self.gridline = self._get_gridline(loc)        

        self.label1 = self._get_text1(loc)
        self.label = self.label1  # legacy name
        self.label2 = self._get_text2(loc)

        self.gridOn = gridOn
        self.tick1On = tick1On
        self.tick2On = tick2On
        self.label1On = label1On
        self.label2On = label2On

    def set_pad(self, val):
        """
Set the tick label pad in points

ACCEPTS: float"""
        self._pad.set(val)

    def get_pad(self, val):
        'Get the value of the tick label pad in points'
        return self._pad.get()
        
    def _get_text1(self, loc):
        'Get the default Text 1 instance'
        pass

    def _get_text2(self, loc):
        'Get the default Text 2 instance'
        pass

    def _get_tick1line(self, loc):
        'Get the default line2D instance for tick1'
        pass

    def _get_tick2line(self, loc):
        'Get the default line2D instance for tick2'
        pass

    def _get_gridline(self, loc):
        'Get the default grid Line2d instance for this tick'
        pass


    def get_loc(self):
        'Return the tick location (data coords) as a scalar'
        return self._loc
    
    def draw(self, renderer):
        if not self.get_visible(): return 
        renderer.open_group(self.__name__)
        midPoint = self.get_view_interval().contains_open( self.get_loc() )

        if midPoint and self.gridOn:  self.gridline.draw(renderer)
        if midPoint and self.tick1On: self.tick1line.draw(renderer)
        if midPoint and self.tick2On: self.tick2line.draw(renderer)


        if self.label1On: self.label1.draw(renderer)
        if self.label2On: self.label2.draw(renderer)

        renderer.close_group(self.__name__)

    def set_xy(self, loc):
        """
Set the location of tick in data coords with scalar loc

ACCEPTS: float"""
        raise NotImplementedError('Derived must override')

    def set_label(self, s):  # legacy name
        """
Set the text of ticklabel

ACCEPTS: str"""
        self.label1.set_text(s)

    def set_label1(self, s):
        """
Set the text of ticklabel

ACCEPTS: str"""
        self.label1.set_text(s)

    def set_label2(self, s):
        """
Set the text of ticklabel2

ACCEPTS: str"""
        self.label2.set_text(s)

    def _set_artist_props(self, a):
        a.set_figure(self.figure)
        #if isinstance(a, Line2D): a.set_clip_box(self.axes.bbox)
        
    def get_view_interval(self):
        'return the view Interval instance for the axis tjis tick is ticking'
        raise NotImplementedError('Derived must override')

        
        
class XTick(Tick):
    """
    Contains all the Artists needed to make an x tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'xtick'
    def _get_text1(self, loc):
        'Get the default Text instance'
        # the y loc is 3 points below the min of y axis
        # get the affine as an a,b,c,d,tx,ty list
        # x in data coords, y in axes coords
        #t =  Text(
        t =  TextWithDash(
            x=loc, y=0,
            fontproperties=FontProperties(size=rcParams['tick.labelsize']),
            color=rcParams['tick.color'],
            verticalalignment='top',
            horizontalalignment='center',
            dashdirection=0,
            xaxis=True,
            )


        trans = blend_xy_sep_transform( self.axes.transData,
                                        self.axes.transAxes)
        #offset the text downward with a post transformation
        transOffset = translation_transform(
            Value(0), Value(-1)*self._padPixels)
        trans.set_offset( (0,0), transOffset)
        t.set_transform( trans)

        self._set_artist_props(t)
        return t


    def _get_text2(self, loc):

        'Get the default Text 2 instance'
        # x in data coords, y in axes coords
        #t =  Text(
        t =  TextWithDash(
            x=loc, y=1,
            fontproperties=FontProperties(size=rcParams['tick.labelsize']),
            color=rcParams['tick.color'],
            verticalalignment='bottom',
            dashdirection=1,
            xaxis=True,
            horizontalalignment='center',
            )

        trans = blend_xy_sep_transform( self.axes.transData,
                                        self.axes.transAxes)
        # offset the text upward with a post transformation
        transOffset = translation_transform(
            Value(0), self._padPixels)
        trans.set_offset( (0,0), transOffset)
        t.set_transform( trans )
        self._set_artist_props(t)
        return t
        
    def _get_tick1line(self, loc): 
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D( xdata=(loc, loc), ydata=(0, 0),
                    color='k',
                    linestyle = 'None',
                    antialiased=False,
                    marker = TICKUP,
                    markersize=self._size,
                    )

        l.set_transform( blend_xy_sep_transform( self.axes.transData,
                                       self.axes.transAxes) )
        self._set_artist_props(l)
        return l

    def _get_tick2line(self, loc):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D( xdata=(loc, loc), ydata=(1,1),
                       color='k',
                       linestyle = 'None',
                       antialiased=False, 
                       marker = TICKDOWN,
                       markersize=self._size,
                       )

        l.set_transform( blend_xy_sep_transform( self.axes.transData,
                                                 self.axes.transAxes) )
        self._set_artist_props(l)
        return l

    def _get_gridline(self, loc):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D( xdata=(loc, loc), ydata=(0, 1),
                    color=rcParams['grid.color'],
                    linestyle=rcParams['grid.linestyle'],
                    linewidth=rcParams['grid.linewidth'],                    
                    antialiased=False, 
                    )
        l.set_transform( blend_xy_sep_transform( self.axes.transData,
                                                 self.axes.transAxes) )
        l.set_clip_box(self.axes.bbox)
        self._set_artist_props(l)

        return l

    def update_position(self, loc):
        'Set the location of tick in data coords with scalar loc'
        x = loc


        self.tick1line.set_xdata((x, x))
        self.tick2line.set_xdata((x, x))
        self.gridline.set_xdata((x, x))
        self.label1.set_x(x)
        self.label2.set_x(x)
        self._loc = loc

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervalx()

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervalx()

        
class YTick(Tick):
    """
    Contains all the Artists needed to make a Y tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'ytick'
    
    # how far from the y axis line the right of the ticklabel are
    def _get_text1(self, loc):
        'Get the default Text instance'
        # x in axes coords, y in data coords
        #t =  Text(
        t =  TextWithDash(
            x=0, y=loc,
            fontproperties=FontProperties(size=rcParams['tick.labelsize']),
            color=rcParams['tick.color'],
            verticalalignment='center',
            horizontalalignment='right',
            dashdirection=0,
            xaxis=False,
            )
        trans = blend_xy_sep_transform( self.axes.transAxes,
                                        self.axes.transData)
        # offset the text leftward with a post transformation

        transOffset = translation_transform(
            Value(-1)*self._padPixels, Value(0))
        trans.set_offset( (0,0), transOffset)
        t.set_transform( trans )
        #t.set_transform( self.axes.transData )
        self._set_artist_props(t)
        return t
    
    def _get_text2(self, loc):
        'Get the default Text instance'
        # x in axes coords, y in data coords
        #t =  Text(
        t =  TextWithDash(
            x=1, y=loc,
            fontproperties=FontProperties(size=rcParams['tick.labelsize']),
            color=rcParams['tick.color'],
            verticalalignment='center',
            dashdirection=1,
            xaxis=False,
            horizontalalignment='left',
            )
        trans = blend_xy_sep_transform( self.axes.transAxes,
                                        self.axes.transData)
        # offset the text rightward with a post transformation

        transOffset = translation_transform(
            self._padPixels, Value(0))
        trans.set_offset( (0,0), transOffset)
        t.set_transform( trans )
        self._set_artist_props(t)
        return t

    def _get_tick1line(self, loc): 
        'Get the default line2D instance'
        # x in axes coords, y in data coords

        l = Line2D( (0, 0), (loc, loc), color='k',
                    antialiased=False,
                    marker = TICKRIGHT,
                    linestyle = 'None',
                    markersize=self._size,
                       )
        l.set_transform( blend_xy_sep_transform( self.axes.transAxes,
                                       self.axes.transData) )
        self._set_artist_props(l)
        return l
    
    def _get_tick2line(self, loc):
        'Get the default line2D instance'
        # x in axes coords, y in data coords
        l = Line2D( (1, 1), (0, 0), color='k',
                    antialiased=False, 
                    marker = TICKLEFT,
                    linestyle = 'None',
                    markersize=self._size,
                    )

        l.set_transform( blend_xy_sep_transform( self.axes.transAxes,
                                                 self.axes.transData) )
        self._set_artist_props(l)
        return l
    
    def _get_gridline(self, loc): 
        'Get the default line2D instance'
        # x in axes coords, y in data coords
        l = Line2D( xdata=(0,1), ydata=(loc,loc), 
                    color=rcParams['grid.color'],
                    linestyle=rcParams['grid.linestyle'],
                    linewidth=rcParams['grid.linewidth'],
                    antialiased=False, 
                    )

        l.set_transform( blend_xy_sep_transform( self.axes.transAxes,
                                                 self.axes.transData) )
        l.set_clip_box(self.axes.bbox)
        
        self._set_artist_props(l)
        return l


    def update_position(self, loc):
        'Set the location of tick in data coords with scalar loc'
        y = loc
        self.tick1line.set_ydata((y, y))
        self.tick2line.set_ydata((y, y))
        self.gridline.set_ydata((y, y))

        self.label1.set_y( y )
        self.label2.set_y( y ) 

        self._loc = loc


    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervaly()

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervaly()


class Ticker:
    locator = None
    formatter = None


class Axis(Artist):

    """
    Public attributes
      transData - transform data coords to display coords
      transAxis - transform axis coords to display coords

    """
    LABELPAD = 5
    OFFSETTEXTPAD = 3
    
    def __init__(self, axes):
        """
        Init the axis with the parent Axes instance
        """
        Artist.__init__(self)
        self.set_figure(axes.figure)

        self.axes = axes
        self.major = Ticker()
        self.minor = Ticker()
        #class dummy:
        #    locator = None
        #    formatter = None
        #self.major = dummy()
        #self.minor = dummy()

        self.label = self._get_label()
        self.offsetText = self._get_offset_text()
        self.majorTicks = []
        self.minorTicks = []

        self.cla()
        
    def cla(self):
        'clear the current axis'
        self.set_major_locator(AutoLocator())
        self.set_major_formatter(ScalarFormatter())
        self.set_minor_locator(NullLocator())
        self.set_minor_formatter(NullFormatter())
        
        # whether the grids are on
        self._gridOnMajor = rcParams['axes.grid']  
        self._gridOnMinor = False
        
        self.label.set_text('')
        self._set_artist_props(self.label)

        # build a few default ticks; grow as necessary later; only
        # define 1 so properties set on ticks will be copied as they
        # grow

        popall(self.majorTicks)
        popall(self.minorTicks)        

        self.majorTicks.extend([self._get_tick(major=True)  for i in range(1)])
        self.minorTicks.extend([self._get_tick(major=False) for i in range(1)])
        
    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        raise NotImplementedError('Derived must override')

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        raise NotImplementedError('Derived must override')

    def _set_artist_props(self, a):
        if a is None: return 
        a.set_figure(self.figure)

    def draw(self, renderer, *args, **kwargs):
        'Draw the axis lines, grid lines, tick lines and labels'
        if not self.get_visible(): return 
        renderer.open_group(__name__)
        ticklabelBoxes = []
        ticklabelBoxes2 = []
        
        majorTicks = self.get_major_ticks()
        majorLocs = self.major.locator()
        self.major.formatter.set_locs(majorLocs)
        majorLabels = [self.major.formatter(val, i) for i, val in enumerate(majorLocs)]


        seen = {}

        interval = self.get_view_interval()
        for tick, loc, label in zip(majorTicks, majorLocs, majorLabels):
            if tick is None: continue
            if not interval.contains(loc): continue
            seen[loc] = 1
            tick.update_position(loc)
            tick.set_label1(label)
            tick.set_label2(label)
            tick.draw(renderer)
            if tick.label1On:
                extent = tick.label1.get_window_extent(renderer) 
                ticklabelBoxes.append(extent)
            if tick.label2On:
                extent = tick.label2.get_window_extent(renderer) 
                ticklabelBoxes2.append(extent)

        minorTicks = self.get_minor_ticks()
        minorLocs = self.minor.locator()
        self.minor.formatter.set_locs(minorLocs)
        minorLabels = [self.minor.formatter(val, i) for i, val in enumerate(minorLocs)]

        for tick, loc, label in zip(minorTicks, minorLocs, minorLabels):
            if tick is None: continue
            if not interval.contains(loc): continue
            if seen.has_key(loc): continue
            tick.update_position(loc)
            tick.set_label1(label)
            tick.set_label2(label)            

            tick.draw(renderer)  
            if tick.label1On:
                extent = tick.label1.get_window_extent(renderer) 
                ticklabelBoxes.append(extent)
            if tick.label2On:
                extent = tick.label2.get_window_extent(renderer) 
                ticklabelBoxes2.append(extent)

        # scale up the axis label box to also find the neighbors, not
        # just the tick labels that actually overlap note we need a
        # *copy* of the axis label box because we don't wan't to scale
        # the actual bbox
        self._update_label_position(ticklabelBoxes, ticklabelBoxes2)
        self.label.draw(renderer)  # memory leak here, vertical text
        
        self._update_offset_text_position(ticklabelBoxes, ticklabelBoxes2)
        self.offsetText.set_text( self.major.formatter.get_offset() )
        self.offsetText.draw(renderer)

        if 0: # draw the bounding boxes around the text for debug
            for tick in majorTicks:
                label = tick.label1
                bbox_artist(label, renderer)
            bbox_artist(self.label, renderer)

        renderer.close_group(__name__)

    def _get_label(self):
        raise NotImplementedError('Derived must override')
        
    def _get_offset_text(self):
        raise NotImplementedError('Derived must override')

    def get_gridlines(self):
        'Return the grid lines as a list of Line2D instance'
        return silent_list('Line2D gridline', [tick.gridline for tick in self.majorTicks])

    def get_label(self):
        'Return the axis label as a Text instance'
        return self.label
        
    def get_offset_text(self):
        'Return the axis offsetText as a Text instance'
        return self.offsetText

    def get_ticklabels(self):
        'Return a list of Text instances for ticklabels'
        labels1 = [tick.label1 for tick in self.get_major_ticks() if tick.label1On]
        labels2 = [tick.label2 for tick in self.get_major_ticks() if tick.label2On]
        return silent_list('Text ticklabel', labels1+labels2)
        
    def get_ticklines(self):
        'Return the ticklines lines as a list of Line2D instance'
        lines = []
        for tick in self.majorTicks:
            lines.append(tick.tick1line)
            lines.append(tick.tick2line)
        return silent_list('Line2D ticklines', lines)

    def get_ticklocs(self):
        "Get the tick locations in data coordinates as a Numeric array"
        return self.major.locator()

    def _get_tick(self, major):
        'return the default tick intsance'
        raise NotImplementedError('derived must override')

    def _copy_tick_props(self, src, dest):
        'Copy the props from src tick to dest tick'
        if src is None or dest is None: return 
        dest.label1.update_from(src.label1)
        dest.label2.update_from(src.label2)

        dest.tick1line.update_from(src.tick1line)
        dest.tick2line.update_from(src.tick2line)        
        dest.gridline.update_from(src.gridline)

        dest.tick1On = src.tick1On
        dest.tick2On = src.tick2On
        dest.label1On = src.label1On
        dest.label2On = src.label2On

    def get_major_locator(self):
        'Get the locator of the major ticker'
        return self.major.locator 

    def get_minor_locator(self):
        'Get the locator of the minor ticker'
        return self.minor.locator 

    def get_major_formatter(self):
        'Get the formatter of the major ticker'
        return self.major.formatter 

    def get_minor_formatter(self):
        'Get the formatter of the minor ticker'
        return self.minor.formatter 
        
    def get_major_ticks(self):
        'get the tick instances; grow as necessary'

        numticks = len(self.major.locator())

        if len(self.majorTicks)<numticks:
            # update the new tick label properties from the old
            protoTick = self.majorTicks[0]
            for i in range(numticks-len(self.majorTicks)):
                tick = self._get_tick(major=True)
                #tick = protoTick
                if self._gridOnMajor: tick.gridOn = True
                self._copy_tick_props(protoTick, tick)
                self.majorTicks.append(tick)
        ticks = self.majorTicks[:numticks]

        return ticks


    def get_minor_ticks(self):
        'get the minor tick instances; grow as necessary'
        numticks = len(self.minor.locator())
        if len(self.minorTicks)<numticks:
            protoTick = self.minorTicks[0]
            for i in range(numticks-len(self.minorTicks)):
                tick = self._get_tick(major=False)
                if self._gridOnMinor: tick.gridOn = True
                self._copy_tick_props(protoTick, tick)
                self.minorTicks.append(tick)
        ticks = self.minorTicks[:numticks]

        return ticks
        

    def grid(self, b=None, which='major'): 
        """
        Set the axis grid on or off; b is a boolean use which =
        'major' | 'minor' to set the grid for major or minor ticks

        if b is None, toggle the grid state
        """
        if which.lower().find('minor')>=0:
            if b is None: self._gridOnMinor = not self._gridOnMinor
            else: self._gridOnMinor = b
            for tick in self.minorTicks:  # don't use get_ticks here!
                if tick is None: continue
                tick.gridOn = self._gridOnMinor
        else:
            if b is None: self._gridOnMajor = not self._gridOnMajor
            else: self._gridOnMajor = b
            for tick in self.majorTicks:  # don't use get_ticks here!
                if tick is None: continue
                tick.gridOn = self._gridOnMajor
                
    
    def set_major_formatter(self, formatter):
        """
Set the formatter of the major ticker

ACCEPTS: A Formatter instance"""
        self.major.formatter = formatter
        self.major.formatter.set_view_interval( self.get_view_interval() )
        self.major.formatter.set_data_interval( self.get_data_interval() )

    def set_minor_formatter(self, formatter):
        """
Set the formatter of the minor ticker

ACCEPTS: A Formatter instance"""
        self.minor.formatter = formatter
        self.minor.formatter.set_view_interval( self.get_view_interval() )
        self.minor.formatter.set_data_interval( self.get_data_interval() )


    def set_major_locator(self, locator):
        """
Set the locator of the major ticker

ACCEPTS: a Locator instance"""
        self.major.locator = locator
        self.major.locator.set_view_interval( self.get_view_interval() )
        self.major.locator.set_data_interval( self.get_data_interval() )


    def set_minor_locator(self, locator):
        """
Set the locator of the minor ticker

ACCEPTS: a Locator instance"""
        self.minor.locator = locator
        self.minor.locator.set_view_interval( self.get_view_interval() )
        self.minor.locator.set_data_interval( self.get_data_interval() )
        
               
    def set_ticklabels(self, ticklabels, *args, **kwargs):
        """
Set the text values of the tick labels. Return a list of Text
instances.

ACCEPTS: sequence of strings"""
        #ticklabels = [str(l) for l in ticklabels]

        self.set_major_formatter( FixedFormatter(ticklabels) )

    
        ret = []
        for i, tick in enumerate(self.get_major_ticks()):
            if i<len(ticklabels): ret.append(tick.label1)
            tick.label1.update(kwargs)
        return ret
    
    def set_ticks(self, ticks):
        """
Set the locations of the tick marks from sequence ticks

ACCEPTS: sequence of floats"""
        self.set_major_locator( FixedLocator(ticks) )

        self.get_view_interval().update(ticks,0)
        return self.get_major_ticks()
    
    def _update_label_position(self, bboxes, bboxes2):
        """
        Update the label position based on the sequence of bounding
        boxes of all the ticklabels
        """
        raise NotImplementedError('Derived must override')
        
    def _update_offset_text_postion(self, bboxes, bboxes2):
        """
        Update the label position based on the sequence of bounding
        boxes of all the ticklabels
        """
        raise NotImplementedError('Derived must override')

    def pan(self, numsteps):
        'Pan numticks (can be positive or negative)'
        self.major.locator.pan(numsteps)

    def zoom(self, direction):
        "Zoom in/out on axis; if direction is >0 zoom in, else zoom out"
        self.major.locator.zoom(direction)
        
class XAxis(Axis):
    __name__ = 'xaxis'

    def _get_tick(self, major):
        return XTick(self.axes, 0, '', major=major)

    def _get_label(self):
        # x in axes coords, y in display coords (to be updated at draw
        # time by _update_label_positions)
        label = Text(x=0.5, y=0,  
            fontproperties = FontProperties(size=rcParams['axes.labelsize']),
            color = rcParams['axes.labelcolor'],
            verticalalignment='top',
            horizontalalignment='center',
            )
        label.set_transform( blend_xy_sep_transform( self.axes.transAxes,
                                                     identity_transform() ))

        self._set_artist_props(label)
        self.label_position='bottom'
        return label
        
    def _get_offset_text(self):
        # x in axes coords, y in display coords (to be updated at draw time)
        offsetText = Text(x=1, y=0,
            fontproperties = FontProperties(size=rcParams['tick.labelsize']),
            color = rcParams['tick.color'],
            verticalalignment='top',
            horizontalalignment='right',
            )
        offsetText.set_transform( blend_xy_sep_transform( self.axes.transAxes,
                                                     identity_transform() ))
        self._set_artist_props(offsetText)
        self.offset_text_position='bottom'
        return offsetText

    def get_label_position(self):
        """
Return the label position (top or bottom)
        """
        return self.label_position

    def set_label_position(self, position):
        """
Set the label position (top or bottom)

ACCEPTS: [ 'top' | 'bottom' ]
        """
        assert position == 'top' or position == 'bottom'
        if position == 'top':
            self.label.set_verticalalignment('bottom')
        else:
            self.label.set_verticalalignment('top')
        self.label_position=position

    def _update_label_position(self, bboxes, bboxes2):
        """
        Update the label position based on the sequence of bounding
        boxes of all the ticklabels 
        """

        x,y = self.label.get_position()
        if self.label_position == 'bottom':
            if not len(bboxes):
                bottom = self.axes.bbox.ymin()
            else:

                bbox = bbox_all(bboxes)
                bottom = bbox.ymin()
            
            self.label.set_position( (x, bottom-self.LABELPAD*self.figure.dpi.get()/72.0))

        else:
            if not len(bboxes2):
                top = self.axes.bbox.ymax()
            else:

                bbox = bbox_all(bboxes2)
                top = bbox.ymax()
            
            self.label.set_position( (x, top+self.LABELPAD*self.figure.dpi.get()/72.0))
            
    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels 
        """
        x,y = self.offsetText.get_position()
        if not len(bboxes):
            bottom = self.axes.bbox.ymin()
        else:
            bbox = bbox_all(bboxes)
            bottom = bbox.ymin()
        self.offsetText.set_position((x, bottom-self.OFFSETTEXTPAD*self.figure.dpi.get()/72.0))

    def tick_top(self):
        'use ticks only on top'
        ticks = self.get_major_ticks()
        ticks.extend( self.get_minor_ticks() )
        for t in ticks:
            t.tick1On = False
            t.tick2On = True
            t.label1On = False
            t.label2On = True

    def tick_bottom(self):
        'use ticks only on bottom'
        ticks = self.get_major_ticks()
        ticks.extend( self.get_minor_ticks() )
        for t in ticks:
            t.tick1On = True
            t.tick2On = False
            t.label1On = True
            t.label2On = False

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervalx()

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervalx()


class YAxis(Axis):
    __name__ = 'yaxis'

    def _get_tick(self, major):
        return YTick(self.axes, 0, '', major=major)


    def _get_label(self):
        # x in display coords (updated by _update_label_position)
        # y in axes coords
        label = Text(x=0, y=0.5,
            # todo: get the label position
            fontproperties=FontProperties(size=rcParams['axes.labelsize']),
            color    = rcParams['axes.labelcolor'],
            verticalalignment='center',
            horizontalalignment='right',
            rotation='vertical', 
            )
        label.set_transform( blend_xy_sep_transform( identity_transform(),
                                                     self.axes.transAxes) )

        self._set_artist_props(label)
        self.label_position='left'
        return label
        
    def _get_offset_text(self):
        # x in display coords, y in axes coords (to be updated at draw time)
        offsetText = Text(x=0, y=0.5,
            fontproperties = FontProperties(size=rcParams['tick.labelsize']),
            color = rcParams['tick.color'],
            verticalalignment='bottom',
            horizontalalignment='left',
            )
        offsetText.set_transform(blend_xy_sep_transform(self.axes.transAxes,
                                                        identity_transform()) )
        self._set_artist_props(offsetText)
        self.offset_text_position='left'
        return offsetText

    def get_label_position(self):
        """
Return the label position (left or right)
        """
        return self.label_position

    def set_label_position(self, position):
        """
Set the label position (left or right)

ACCEPTS: [ 'left' | 'right' ]
        """
        assert position == 'left' or position == 'right'
        if position == 'right':
            self.label.set_horizontalalignment('left')
        else:
            self.label.set_horizontalalignment('right')
        self.label_position=position

    def _update_label_position(self, bboxes, bboxes2):
        """
        Update the label position based on the sequence of bounding
        boxes of all the ticklabels 
        """

        x,y = self.label.get_position()
        if self.label_position == 'left':
            if not len(bboxes):
                left = self.axes.bbox.xmin()
            else:

                bbox = bbox_all(bboxes)
                left = bbox.xmin()
            
            self.label.set_position( (left-self.LABELPAD*self.figure.dpi.get()/72.0, y))

        else:
            if not len(bboxes2):
                right = self.axes.bbox.xmax()
            else:

                bbox = bbox_all(bboxes2)
                right = bbox.xmax()
            
            self.label.set_position( (right+self.LABELPAD*self.figure.dpi.get()/72.0, y))
            
    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels 
        """
        x,y = self.offsetText.get_position()
        top = self.axes.bbox.ymax()
        self.offsetText.set_position((x, top+self.OFFSETTEXTPAD*self.figure.dpi.get()/72.0))

    def tick_right(self):
        'use ticks only on right'
        ticks = self.get_major_ticks()
        ticks.extend( self.get_minor_ticks() )
        for t in ticks:
            t.tick1On = False
            t.tick2On = True
            t.label1On = False
            t.label2On = True

    def tick_left(self):
        'use ticks only on left'
        ticks = self.get_major_ticks()
        ticks.extend( self.get_minor_ticks() )
        for t in ticks:
            t.tick1On = True
            t.tick2On = False
            t.label1On = True
            t.label2On = False

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervaly()

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervaly()



