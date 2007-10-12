"""
Classes for the ticks and x and y axis
"""
from __future__ import division
import copy
import math
import re
import sys

from artist import Artist, setp
from cbook import enumerate, silent_list, popall, CallbackRegistry
from lines import Line2D, TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN
from matplotlib import rcParams
from patches import bbox_artist
from ticker import NullFormatter, FixedFormatter, ScalarFormatter, LogFormatter, LogFormatterMathtext
from ticker import NullLocator, FixedLocator, LinearLocator, LogLocator, AutoLocator

from font_manager import FontProperties
from text import Text, TextWithDash, _process_text_args
from transforms import Affine2D, Bbox, blended_transform_factory, interval_contains, \
    interval_contains_open, IntervalTransform, IdentityTransform
from patches import bbox_artist
from scale import scale_factory

import matplotlib.units as units


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

        name = self.__name__.lower()
        if size is None:
            if major:
                size = rcParams['%s.major.size'%name]
                pad = rcParams['%s.major.pad'%name]
            else:
                size = rcParams['%s.minor.size'%name]
                pad = rcParams['%s.minor.pad'%name]

        self._tickdir = rcParams['%s.direction'%name]
        if self._tickdir == 'in':
            self._xtickmarkers = (TICKUP, TICKDOWN)
            self._ytickmarkers = (TICKRIGHT, TICKLEFT)
            self._pad = pad
        else:
            self._xtickmarkers = (TICKDOWN, TICKUP)
            self._ytickmarkers = (TICKLEFT, TICKRIGHT)
            self._pad = pad + size

        self._loc = loc
        self._size = size

        self._padPixels = self.figure.dpi * self._pad * (1/72.0)
        self._locTransform = Affine2D()
        
        self.tick1line = self._get_tick1line()
        self.tick2line = self._get_tick2line()
        self.gridline = self._get_gridline()

        self.label1 = self._get_text1()
        self.label = self.label1  # legacy name
        self.label2 = self._get_text2()

        self.update_position(loc)
        
        self.gridOn = gridOn
        self.tick1On = tick1On
        self.tick2On = tick2On
        self.label1On = label1On
        self.label2On = label2On

    def get_children(self):
        children = [self.tick1line, self.tick2line, self.gridline, self.label1, self.label2]
        return children

    def set_clip_path(self, clippath, transform=None):
        Artist.set_clip_path(self, clippath, transform)
        self.tick1line.set_clip_path(clippath, transform)
        self.tick2line.set_clip_path(clippath, transform)
        self.gridline.set_clip_path(clippath, transform)
    
    def contains(self, mouseevent):
        """Test whether the mouse event occured in the Tick marks.

        This function always returns false.  It is more useful to test if the
        axis as a whole contains the mouse rather than the set of tick marks.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        return False,{}

    def set_pad(self, val):
        """
        Set the tick label pad in points

        ACCEPTS: float
        """
        self._pad.set(val)

    def get_pad(self, val):
        'Get the value of the tick label pad in points'
        return self._pad.get()

    def _get_text1(self):
        'Get the default Text 1 instance'
        pass

    def _get_text2(self):
        'Get the default Text 2 instance'
        pass

    def _get_tick1line(self):
        'Get the default line2D instance for tick1'
        pass

    def _get_tick2line(self):
        'Get the default line2D instance for tick2'
        pass

    def _get_gridline(self):
        'Get the default grid Line2d instance for this tick'
        pass


    def get_loc(self):
        'Return the tick location (data coords) as a scalar'
        return self._loc

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group(self.__name__)
        midPoint = interval_contains(self.get_view_interval(), self.get_loc())

        if midPoint:
            if self.gridOn:
                self.gridline.draw(renderer)
            if self.tick1On:
                self.tick1line.draw(renderer)
            if self.tick2On:
                self.tick2line.draw(renderer)

        if self.label1On:
            self.label1.draw(renderer)
        if self.label2On:
            self.label2.draw(renderer)

        renderer.close_group(self.__name__)

    def set_label1(self, s):
        """
        Set the text of ticklabel

        ACCEPTS: str
        """
        self.label1.set_text(s)
    set_label = set_label1
        
    def set_label2(self, s):
        """
        Set the text of ticklabel2

        ACCEPTS: str
        """
        self.label2.set_text(s)

    def _set_artist_props(self, a):
        a.set_figure(self.figure)
        #if isinstance(a, Line2D): a.set_clip_box(self.axes.bbox)

    def get_view_interval(self):
        'return the view Interval instance for the axis tjis tick is ticking'
        raise NotImplementedError('Derived must override')

    def set_view_interval(self, vmin, vmax, ignore=False):
        raise NotImplementedError('Derived must override')

    
class XTick(Tick):
    """
    Contains all the Artists needed to make an x tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'xtick'
    def _get_text1(self):
        'Get the default Text instance'
        # the y loc is 3 points below the min of y axis
        # get the affine as an a,b,c,d,tx,ty list
        # x in data coords, y in axes coords
        #t =  Text(
        trans, vert, horiz = self.axes.get_xaxis_text1_transform(self._padPixels)
        
        t =  TextWithDash(
            x=0, y=0,
            fontproperties=FontProperties(size=rcParams['xtick.labelsize']),
            color=rcParams['xtick.color'],
            verticalalignment=vert,
            horizontalalignment=horiz,
            dashdirection=0,
            xaxis=True,
            )

        t.set_transform(trans)
        self._set_artist_props(t)
        return t


    def _get_text2(self):

        'Get the default Text 2 instance'
        # x in data coords, y in axes coords
        #t =  Text(
        trans, vert, horiz = self.axes.get_xaxis_text2_transform(self._padPixels)

        t = TextWithDash(
            x=0, y=1,
            fontproperties=FontProperties(size=rcParams['xtick.labelsize']),
            color=rcParams['xtick.color'],
            verticalalignment=vert,
            dashdirection=1,
            xaxis=True,
            horizontalalignment=horiz,
            )
        t.set_transform(trans)
        self._set_artist_props(t)
        return t

    def _get_tick1line(self):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D(xdata=(0,), ydata=(0,),
                   color='k',
                   linestyle = 'None',
                   marker = self._xtickmarkers[0],
                   markersize=self._size,
                   )
        l.set_transform(self.axes.get_xaxis_transform())
        self._set_artist_props(l)
        return l

    def _get_tick2line(self):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D( xdata=(0,), ydata=(1,),
                       color='k',
                       linestyle = 'None',
                       marker = self._xtickmarkers[1],
                       markersize=self._size,
                       )

        l.set_transform(self._locTransform + self.axes.get_xaxis_transform())
        self._set_artist_props(l)
        return l

    def _get_gridline(self):
        'Get the default line2D instance'
        # x in data coords, y in axes coords
        l = Line2D(xdata=(0.0, 0.0), ydata=(0, 1.0),
                   color=rcParams['grid.color'],
                   linestyle=rcParams['grid.linestyle'],
                   linewidth=rcParams['grid.linewidth'],
                   )
        l.set_transform(self._locTransform + self.axes.get_xaxis_transform())
        self._set_artist_props(l)

        return l

    def update_position(self, loc):
        'Set the location of tick in data coords with scalar loc'
        self._locTransform.clear().translate(loc, 0.0)
        self.label1.set_x(loc)
        self.label2.set_x(loc)
        self._loc = loc

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervalx

    def set_view_interval(self, vmin, vmax, ignore = False):
        if ignore:
            self.axes.viewLim.intervalx = vmin, vmax
        else:
            Vmin, Vmax = self.get_view_interval()
            self.axes.viewLim.intervalx = min(vmin, Vmin), max(vmax, Vmax)
    
    def get_minpos(self):
        return self.axes.dataLim.minposx
    
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervalx


class YTick(Tick):
    """
    Contains all the Artists needed to make a Y tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'ytick'

    # how far from the y axis line the right of the ticklabel are
    def _get_text1(self):
        'Get the default Text instance'
        # x in axes coords, y in data coords
        #t =  Text(
        trans, vert, horiz = self.axes.get_yaxis_text1_transform(self._padPixels)

        t = TextWithDash(
            x=0, y=0,
            fontproperties=FontProperties(size=rcParams['ytick.labelsize']),
            color=rcParams['ytick.color'],
            verticalalignment=vert,
            horizontalalignment=horiz,
            dashdirection=0,
            xaxis=False,
            )
        t.set_transform(trans)
        #t.set_transform( self.axes.transData )
        self._set_artist_props(t)
        return t

    def _get_text2(self):
        'Get the default Text instance'
        # x in axes coords, y in data coords
        #t =  Text(
        trans, vert, horiz = self.axes.get_yaxis_text2_transform(self._padPixels)

        t = TextWithDash(
            x=1, y=0,
            fontproperties=FontProperties(size=rcParams['ytick.labelsize']),
            color=rcParams['ytick.color'],
            verticalalignment=vert,
            dashdirection=1,
            xaxis=False,
            horizontalalignment=horiz,
            )
        t.set_transform(trans)
        self._set_artist_props(t)
        return t

    def _get_tick1line(self):
        'Get the default line2D instance'
        # x in axes coords, y in data coords

        l = Line2D( (0,), (0,), color='k',
                    marker = self._ytickmarkers[0],
                    linestyle = 'None',
                    markersize=self._size,
                       )
        l.set_transform(self._locTransform + self.axes.get_yaxis_transform())
        self._set_artist_props(l)
        return l

    def _get_tick2line(self):
        'Get the default line2D instance'
        # x in axes coords, y in data coords
        l = Line2D( (1,), (0,), color='k',
                    marker = self._ytickmarkers[1],
                    linestyle = 'None',
                    markersize=self._size,
                    )

        l.set_transform(self._locTransform + self.axes.get_yaxis_transform())
        self._set_artist_props(l)
        return l

    def _get_gridline(self):
        'Get the default line2D instance'
        # x in axes coords, y in data coords
        l = Line2D( xdata=(0,1), ydata=(0, 0),
                    color=rcParams['grid.color'],
                    linestyle=rcParams['grid.linestyle'],
                    linewidth=rcParams['grid.linewidth'],
                    )

        l.set_transform(self._locTransform + self.axes.get_yaxis_transform())
        self._set_artist_props(l)
        return l


    def update_position(self, loc):
        'Set the location of tick in data coords with scalar loc'
        self._locTransform.clear().translate(0.0, loc)
        self.label1.set_y(loc)
        self.label2.set_y(loc)
        self._loc = loc


    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervaly

    def set_view_interval(self, vmin, vmax):
        if ignore:
            self.axes.viewLim.intervaly = vmin, vmax
        else:
            Vmin, Vmax = self.get_view_interval()
            self.axes.viewLim.intervaly = min(vmin, Vmin), max(vmax, Vmax)
    
    def get_minpos(self):
        return self.axes.dataLim.minposy
    
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervaly


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
    
    def __str__(self):
        return str(self.__class__).split('.')[-1] \
            + "(%d,%d)"%self.axes.transAxes.xy_tup((0,0))

    def __init__(self, axes, pickradius=15):
        """
        Init the axis with the parent Axes instance
        """
        Artist.__init__(self)
        self.set_figure(axes.figure)

        self.axes = axes
        self.major = Ticker()
        self.minor = Ticker()
        self.callbacks = CallbackRegistry(('units', 'units finalize'))

        #class dummy:
        #    locator = None
        #    formatter = None
        #self.major = dummy()
        #self.minor = dummy()

        self.label = self._get_label()
        self.offsetText = self._get_offset_text()
        self.majorTicks = []
        self.minorTicks = []
        self.pickradius = pickradius
        
        self.cla()
        self.set_scale('linear')
        
    def get_transform(self):
        return self._scale.get_transform()

    def get_scale(self):
        return self._scale.name
    
    def set_scale(self, value, **kwargs):
        self._scale = scale_factory(value, self, **kwargs)
        self._scale.set_default_locators_and_formatters(self)

    def limit_range_for_scale(self, vmin, vmax):
        return self._scale.limit_range_for_scale(vmin, vmax, self.get_minpos())
        
    def get_children(self):
        children = [self.label]
        majorticks = self.get_major_ticks()
        minorticks = self.get_minor_ticks()        
        
        children.extend(majorticks)
        children.extend(minorticks)
        return children

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

        self.majorTicks.extend([self._get_tick(major=True)])
        self.minorTicks.extend([self._get_tick(major=False)])
        self._lastNumMajorTicks = 1
        self._lastNumMinorTicks = 1
        
        self.converter = None
        self.units = None
        self.set_units(None)

    def set_clip_path(self, clippath, transform=None):
        Artist.set_clip_path(self, clippath, transform)
        majorticks = self.get_major_ticks()
        minorticks = self.get_minor_ticks()        
        for child in self.majorTicks + self.minorTicks:
            child.set_clip_path(clippath, transform)
        
    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        raise NotImplementedError('Derived must override')

    def set_view_interval(self, vmin, vmax, ignore=False):
        raise NotImplementedError('Derived must override')
    
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        raise NotImplementedError('Derived must override')

    def set_data_interval(self):
        'Set the axis data limits'
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

        majorLocs = self.major.locator()
        majorTicks = self.get_major_ticks(len(majorLocs))
        self.major.formatter.set_locs(majorLocs)
        majorLabels = [self.major.formatter(val, i) for i, val in enumerate(majorLocs)]


        seen = {}

        interval = self.get_view_interval()
        for tick, loc, label in zip(majorTicks, majorLocs, majorLabels):
            if tick is None: continue
            if not interval_contains(interval, loc): continue
            seen[loc] = 1
            tick.update_position(loc)
            tick.set_label1(label)
            tick.set_label2(label)
            tick.draw(renderer)
            if tick.label1On and tick.label1.get_visible():
                extent = tick.label1.get_window_extent(renderer)
                ticklabelBoxes.append(extent)
            if tick.label2On and tick.label2.get_visible():
                extent = tick.label2.get_window_extent(renderer)
                ticklabelBoxes2.append(extent)

        minorLocs = self.minor.locator()
        minorTicks = self.get_minor_ticks(len(minorLocs))
        self.minor.formatter.set_locs(minorLocs)
        minorLabels = [self.minor.formatter(val, i) for i, val in enumerate(minorLocs)]

        for tick, loc, label in zip(minorTicks, minorLocs, minorLabels):
            if tick is None: continue
            if not interval_contains(interval, loc): continue
            #if seen.has_key(loc): continue
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
        self.label.draw(renderer)

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
        ticks = self.get_major_ticks()
        return silent_list('Line2D gridline', [tick.gridline for tick in ticks])

    def get_label(self):
        'Return the axis label as a Text instance'
        return self.label

    def get_offset_text(self):
        'Return the axis offsetText as a Text instance'
        return self.offsetText

    def get_pickradius(self):
        'Return the depth of the axis used by the picker'
        return self.pickradius

    def get_ticklabels(self):
        'Return a list of Text instances for ticklabels'
        ticks = self.get_major_ticks()
        labels1 = [tick.label1 for tick in ticks if tick.label1On]
        labels2 = [tick.label2 for tick in ticks if tick.label2On]
        return silent_list('Text ticklabel', labels1+labels2)

    def get_ticklines(self):
        'Return the ticklines lines as a list of Line2D instance'
        lines = []
        ticks = self.get_major_ticks()        
        for tick in ticks:
            lines.append(tick.tick1line)
            lines.append(tick.tick2line)
        return silent_list('Line2D ticklines', lines)

    def get_ticklocs(self):
        "Get the tick locations in data coordinates as a numpy array"
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

    def get_major_ticks(self, numticks=None):
        'get the tick instances; grow as necessary'
        if numticks is None:
            numticks = len(self.get_major_locator()())

        if len(self.majorTicks) < numticks:
            # update the new tick label properties from the old
            for i in range(numticks - len(self.majorTicks)):
                tick = self._get_tick(major=True)
                self.majorTicks.append(tick)
            
        if self._lastNumMajorTicks < numticks:
            protoTick = self.majorTicks[0]
            for i in range(self._lastNumMajorTicks, len(self.majorTicks)):
                tick = self.majorTicks[i]
                if self._gridOnMajor: tick.gridOn = True
                self._copy_tick_props(protoTick, tick)

        self._lastNumMajorTicks = numticks
        ticks = self.majorTicks[:numticks]

        return ticks


    def get_minor_ticks(self, numticks=None):
        'get the minor tick instances; grow as necessary'
        if numticks is None:
            numticks = len(self.get_minor_locator()())

        if len(self.minorTicks) < numticks:
            # update the new tick label properties from the old
            for i in range(numticks - len(self.minorTicks)):
                tick = self._get_tick(minor=True)
                self.minorTicks.append(tick)
            
        if self._lastNumMinorTicks < numticks:
            protoTick = self.minorTicks[0]
            for i in range(self._lastNumMinorTicks, len(self.minorTicks)):
                tick = self.minorTicks[i]
                if self._gridOnMinor: tick.gridOn = True
                self._copy_tick_props(protoTick, tick)

        self._lastNumMinorTicks = numticks
        ticks = self.minorTicks[:numticks]

        return ticks


    def grid(self, b=None, which='major', **kwargs):
        """
        Set the axis grid on or off; b is a boolean use which =
        'major' | 'minor' to set the grid for major or minor ticks

        if b is None and len(kwargs)==0, toggle the grid state.  If
        kwargs are supplied, it is assumed you want the grid on and b
        will be set to True

        kwargs are used to set the line properties of the grids, eg,

          xax.grid(color='r', linestyle='-', linewidth=2)
        """
        if len(kwargs): b = True
        if which.lower().find('minor')>=0:
            if b is None: self._gridOnMinor = not self._gridOnMinor
            else: self._gridOnMinor = b
            for tick in self.minorTicks:  # don't use get_ticks here!
                if tick is None: continue
                tick.gridOn = self._gridOnMinor
                if len(kwargs): setp(tick.gridline,**kwargs)
        else:
            if b is None: self._gridOnMajor = not self._gridOnMajor
            else: self._gridOnMajor = b
            for tick in self.majorTicks:  # don't use get_ticks here!
                if tick is None: continue
                tick.gridOn = self._gridOnMajor
                if len(kwargs): setp(tick.gridline,**kwargs)


    def update_units(self, data):
        """
        introspect data for units converter and update the
        axis.converter instance if necessary. Return true is data is
        registered for unit conversion
        """

        converter = units.registry.get_converter(data)
        if converter is None: return False
        self.converter = converter
        default = self.converter.default_units(data)
        #print 'update units: default="%s", units=%s"'%(default, self.units)
        if default is not None and self.units is None:
            self.set_units(default)
        self._update_axisinfo()
        return True

    def _update_axisinfo(self):
        """
        check the axis converter for the stored units to see if the
        axis info needs to be updated
        """

        if self.converter is None:
            return

        info = self.converter.axisinfo(self.units)
        if info is None:
            return
        if info.majloc is not None and self.major.locator!=info.majloc:
            self.set_major_locator(info.majloc)
        if info.minloc is not None and self.minor.locator!=info.minloc:
            self.set_minor_locator(info.minloc)
        if info.majfmt is not None and self.major.formatter!=info.majfmt:
            self.set_major_formatter(info.majfmt)
        if info.minfmt is not None and self.minor.formatter!=info.minfmt:
            self.set_minor_formatter(info.minfmt)
        if info.label is not None:
            label = self.get_label()
            label.set_text(info.label)


    def have_units(self):
        return self.converter is not None or self.units is not None

    def convert_units(self, x):
        if self.converter is None:
            self.converter = units.registry.get_converter(x)

        if self.converter is None:
            #print 'convert_units returning identity: units=%s, converter=%s'%(self.units, self.converter)
            return x

        ret =  self.converter.convert(x, self.units)
        #print 'convert_units converting: axis=%s, units=%s, converter=%s, in=%s, out=%s'%(self, self.units, self.converter, x, ret)
        return ret

    def set_units(self, u):
        """
        set the units for axis

        ACCEPTS: a units tag
        """
        pchanged = False
        if u is None:
            self.units = None
            pchanged = True
        else:
            if u!=self.units:
                self.units = u
                #print 'setting units', self.converter, u, units.registry.get_converter(u)
                pchanged = True
        if pchanged:
            self._update_axisinfo()
            self.callbacks.process('units')
            self.callbacks.process('units finalize')

    def get_units(self):
        'return the units for axis'
        return self.units

    def set_major_formatter(self, formatter):
        """
        Set the formatter of the major ticker

        ACCEPTS: A Formatter instance
        """
        self.major.formatter = formatter
        formatter.set_axis(self)


    def set_minor_formatter(self, formatter):
        """
        Set the formatter of the minor ticker

        ACCEPTS: A Formatter instance
        """
        self.minor.formatter = formatter
        formatter.set_axis(self)


    def set_major_locator(self, locator):
        """
        Set the locator of the major ticker

        ACCEPTS: a Locator instance
        """
        self.major.locator = locator
        locator.set_axis(self)


    def set_minor_locator(self, locator):
        """
        Set the locator of the minor ticker

        ACCEPTS: a Locator instance
        """
        self.minor.locator = locator
        locator.set_axis(self)

    def set_pickradius(self, pickradius):
        """
        Set the depth of the axis used by the picker

        ACCEPTS: a distance in points
        """
        self.pickradius = pickradius


    def set_ticklabels(self, ticklabels, *args, **kwargs):
        """
        Set the text values of the tick labels. Return a list of Text
        instances.

        ACCEPTS: sequence of strings
        """
        #ticklabels = [str(l) for l in ticklabels]

        self.set_major_formatter( FixedFormatter(ticklabels) )

        ret = []
        for i, tick in enumerate(self.get_major_ticks()):
            if i<len(ticklabels):
                tick.label1.set_text(ticklabels[i])
                ret.append(tick.label1)
            tick.label1.update(kwargs)
        return ret

    def set_ticks(self, ticks):
        """
        Set the locations of the tick marks from sequence ticks

        ACCEPTS: sequence of floats
        """
        ### XXX if the user changes units, the information will be lost here
        ticks = self.convert_units(ticks)
        self.set_major_locator( FixedLocator(ticks) )
        if len(ticks):
            self.set_view_interval(min(ticks), max(ticks))
        return self.get_major_ticks(len(ticks))

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
    axis_name = 'x'
            
    def contains(self,mouseevent):
        """Test whether the mouse event occured in the x axis.
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        xpixel,ypixel = mouseevent.x,mouseevent.y
        try:
            xaxes,yaxes = self.axes.transAxes.inverse_xy_tup((xpixel,ypixel))
            xorigin,yorigin = self.axes.transAxes.xy_tup((0,0))
        except ValueError:
            return False, {}
        inaxis = xaxes>=0 and xaxes<=1 and ypixel<yorigin and ypixel>yorigin-self.pickradius
        return inaxis, {}

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
        label.set_transform( blended_transform_factory(
		self.axes.transAxes, Affine2D() ))

        self._set_artist_props(label)
        self.label_position='bottom'
        return label

    def _get_offset_text(self):
        # x in axes coords, y in display coords (to be updated at draw time)
        offsetText = Text(x=1, y=0,
            fontproperties = FontProperties(size=rcParams['xtick.labelsize']),
            color = rcParams['xtick.color'],
            verticalalignment='top',
            horizontalalignment='right',
            )
        offsetText.set_transform( blended_transform_factory(
		self.axes.transAxes, Affine2D() ))
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
                bottom = self.axes.bbox.ymin
            else:
                bbox = Bbox.union(bboxes)
                bottom = bbox.ymin
            self.label.set_position( (x, bottom - self.LABELPAD*self.figure.dpi / 72.0))
            
        else:
            if not len(bboxes2):
                top = self.axes.bbox.ymax
            else:
                bbox = bbox_union(bboxes2)
                top = bbox.ymax
 
            self.label.set_position( (x, top+self.LABELPAD*self.figure.dpi / 72.0))

    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x,y = self.offsetText.get_position()
        if not len(bboxes):
            bottom = self.axes.bbox.ymin
        else:
            bbox = Bbox.union(bboxes)
            bottom = bbox.ymin
        self.offsetText.set_position((x, bottom-self.OFFSETTEXTPAD*self.figure.dpi/72.0))
	
    def set_ticks_position(self, position):
        """
        Set the ticks position (top, bottom, both or default)
        both sets the ticks to appear on both positions, but
        does not change the tick labels.
        default resets the tick positions to the default:
        ticks on both positions, labels at bottom.

        ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]
        """
        assert position in ('top', 'bottom', 'both', 'default', 'none')


        ticks = list( self.get_major_ticks() )  # a copy
        ticks.extend( self.get_minor_ticks() )

        if position == 'top':
            for t in ticks:
                t.tick1On = False
                t.tick2On = True
                t.label1On = False
                t.label2On = True
        elif position == 'bottom':
            for t in ticks:
                t.tick1On = True
                t.tick2On = False
                t.label1On = True
                t.label2On = False
        elif position == 'default':
            for t in ticks:
                t.tick1On = True
                t.tick2On = True
                t.label1On = True
                t.label2On = False
        elif position == 'none':
            for t in ticks:
                t.tick1On = False
                t.tick2On = False
        else:
            for t in ticks:
                t.tick1On = True
                t.tick2On = True

    def tick_top(self):
        'use ticks only on top'
        self.set_ticks_position('top')

    def tick_bottom(self):
        'use ticks only on bottom'
        self.set_ticks_position('bottom')

    def get_ticks_position(self):
        """
        Return the ticks position (top, bottom, default or unknown)
        """
        majt=self.majorTicks[0]
        mT=self.minorTicks[0]

        majorTop=(not majt.tick1On) and majt.tick2On and (not majt.label1On) and majt.label2On
        minorTop=(not mT.tick1On) and mT.tick2On and (not mT.label1On) and mT.label2On
        if majorTop and minorTop: return 'top'

        MajorBottom=majt.tick1On and (not majt.tick2On) and majt.label1On and (not majt.label2On)
        MinorBottom=mT.tick1On and (not mT.tick2On) and mT.label1On and (not mT.label2On)
        if MajorBottom and MinorBottom: return 'bottom'

        majorDefault=majt.tick1On and majt.tick2On and majt.label1On and (not majt.label2On)
        minorDefault=mT.tick1On and mT.tick2On and mT.label1On and (not mT.label2On)
        if majorDefault and minorDefault: return 'default'

        return 'unknown'

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervalx

    def set_view_interval(self, vmin, vmax, ignore=False):
        if ignore:
            self.axes.viewLim.intervalx = vmin, vmax
        else:
            Vmin, Vmax = self.get_view_interval()
            self.axes.viewLim.intervalx = min(vmin, Vmin), max(vmax, Vmax)
    
    def get_minpos(self):
        return self.axes.dataLim.minposx
    
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervalx

    def set_data_interval(self, vmin, vmax, ignore=False):
        'return the Interval instance for this axis data limits'
        if ignore:
            self.axes.dataLim.intervalx = vmin, vmax
        else:
            Vmin, Vmax = self.get_data_interval()
            self.axes.dataLim.intervalx = min(vmin, Vmin), max(vmax, Vmax)


class YAxis(Axis):
    __name__ = 'yaxis'
    axis_name = 'y'
            
    def contains(self,mouseevent):
        """Test whether the mouse event occurred in the y axis.

        Returns T/F, {}
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        xpixel,ypixel = mouseevent.x,mouseevent.y
        try:
            xaxes,yaxes = self.axes.transAxes.inverse_xy_tup((xpixel,ypixel))
            xorigin,yorigin = self.axes.transAxes.xy_tup((0,0))
        except ValueError:
            return False, {}
        inaxis = yaxes>=0 and yaxes<=1 and xpixel<xorigin and xpixel>xorigin-self.pickradius
        return inaxis, {}

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
        label.set_transform( blended_transform_factory(
		Affine2D(), self.axes.transAxes) )

        self._set_artist_props(label)
        self.label_position='left'
        return label

    def _get_offset_text(self):
        # x in display coords, y in axes coords (to be updated at draw time)
        offsetText = Text(x=0, y=0.5,
            fontproperties = FontProperties(size=rcParams['ytick.labelsize']),
            color = rcParams['ytick.color'],
            verticalalignment = 'bottom',
            horizontalalignment = 'left',
            )
        offsetText.set_transform(blended_transform_factory(
		self.axes.transAxes, Affine2D()) )
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
                left = self.axes.bbox.xmin
            else:

                bbox = Bbox.union(bboxes)
                left = bbox.xmin

            self.label.set_position( (left-self.LABELPAD*self.figure.dpi/72.0, y))
	    
        else:
            if not len(bboxes2):
                right = self.axes.bbox.xmax
            else:
                bbox = Bbox.union(bboxes2)
                right = bbox.xmax

            self.label.set_position( (right+self.LABELPAD*self.figure.dpi/72.0, y))

    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x,y = self.offsetText.get_position()
        top = self.axes.bbox.ymax
        self.offsetText.set_position((x, top+self.OFFSETTEXTPAD*self.figure.dpi/72.0))
	
    def set_offset_position(self, position):
        assert position == 'left' or position == 'right'

        x,y = self.offsetText.get_position()
        if position == 'left': x = 0
        else: x = 1

        self.offsetText.set_ha(position)
        self.offsetText.set_position((x,y))

    def set_ticks_position(self, position):
        """
        Set the ticks position (left, right, both or default)
        both sets the ticks to appear on both positions, but
        does not change the tick labels.
        default resets the tick positions to the default:
        ticks on both positions, labels on the left.

        ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]
        """
        assert position in ('left', 'right', 'both', 'default', 'none')

        ticks = list( self.get_major_ticks() ) # a copy
        ticks.extend( self.get_minor_ticks() )

        if position == 'right':
            self.set_offset_position('right')
            for t in ticks:
                t.tick1On = False
                t.tick2On = True
                t.label1On = False
                t.label2On = True
        elif position == 'left':
            self.set_offset_position('left')
            for t in ticks:
                t.tick1On = True
                t.tick2On = False
                t.label1On = True
                t.label2On = False
        elif position == 'default':
            self.set_offset_position('left')
            for t in ticks:
                t.tick1On = True
                t.tick2On = True
                t.label1On = True
                t.label2On = False
        elif position == 'none':
            for t in ticks:
                t.tick1On = False
                t.tick2On = False
        else:
            self.set_offset_position('left')
            for t in ticks:
                t.tick1On = True
                t.tick2On = True

    def tick_right(self):
        'use ticks only on right'
        self.set_ticks_position('right')

    def tick_left(self):
        'use ticks only on left'
        self.set_ticks_position('left')

    def get_ticks_position(self):
        """
        Return the ticks position (left, right, both or unknown)
        """
        majt=self.majorTicks[0]
        mT=self.minorTicks[0]

        majorRight=(not majt.tick1On) and majt.tick2On and (not majt.label1On) and majt.label2On
        minorRight=(not mT.tick1On) and mT.tick2On and (not mT.label1On) and mT.label2On
        if majorRight and minorRight: return 'right'

        majorLeft=majt.tick1On and (not majt.tick2On) and majt.label1On and (not majt.label2On)
        minorLeft=mT.tick1On and (not mT.tick2On) and mT.label1On and (not mT.label2On)
        if majorLeft and minorLeft: return 'left'

        majorDefault=majt.tick1On and majt.tick2On and majt.label1On and (not majt.label2On)
        minorDefault=mT.tick1On and mT.tick2On and mT.label1On and (not mT.label2On)
        if majorDefault and minorDefault: return 'default'

        return 'unknown'

    def get_view_interval(self):
        'return the Interval instance for this axis view limits'
        return self.axes.viewLim.intervaly

    def set_view_interval(self, vmin, vmax, ignore=False):
        if ignore:
            self.axes.viewLim.intervaly = vmin, vmax
        else:
            Vmin, Vmax = self.get_view_interval()
            self.axes.viewLim.intervaly = min(vmin, Vmin), max(vmax, Vmax)
    
    def get_minpos(self):
        return self.axes.dataLim.minposy
    
    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.axes.dataLim.intervaly

    def set_data_interval(self, vmin, vmax, ignore=False):
        'return the Interval instance for this axis data limits'
        if ignore:
            self.axes.dataLim.intervaly = vmin, vmax
        else:
            Vmin, Vmax = self.get_data_interval()
            self.axes.dataLim.intervaly = min(vmin, Vmin), max(vmax, Vmax)


