"""
Tick locating and formatting
============================

This module contains classes to support completely configurable tick
locating and formatting.  Although the locators know nothing about
major or minor ticks, they are used by the Axis class to support major
and minor tick locating and formatting.  Generic tick locators and
formatters are provided, as well as domain specific custom ones..


Tick locating
-------------

The Locator class is the base class for all tick locators.  The
locators handle autoscaling of the view limits based on the data
limits, and the choosing of tick locations.  The most generally useful
tick locator is MultipleLocator.  You initialize this with a base, eg
10, and it picks axis limits and ticks that are multiples of your
base.  The class AutoLocator contains a MultipleLocator instance, and
dynamically updates it based upon the data and zoom limits.  This
should provide much more intelligent automatic tick locations both in
figure creation and in navigation than in prior versions of
matplotlib.

The basic generic  locators are

  * NullLocator     - No ticks

  * IndexLocator    - locator for index plots (eg where x = range(len(y))
  
  * LinearLocator   - evenly spaced ticks from min to max

  * LogLocator      - logarithmically ticks from min to max

  * MultipleLocator - ticks and range are a multiple of base;
                      either integer or float
  
  * AutoLocator     - choose a MultipleLocator and dyamically reassign
                      it for intelligent ticking during navigation
  
There are a number of locators specialized for date locations - see
the dates module

You can define your own locator by deriving from Locator.  You must
override the __call__ method, which returns a sequence of locations,
and you will probably want to override the autoscale method to set the
view limits from the data limits.

If you want to override the default locator, use one of the above or a
custom locator and pass it to the x or y axis instance.  The relevant
methods are::

  ax.xaxis.set_major_locator( xmajorLocator )
  ax.xaxis.set_minor_locator( xminorLocator )
  ax.yaxis.set_major_locator( ymajorLocator )
  ax.yaxis.set_minor_locator( yminorLocator )

The default minor locator is the NullLocator, eg no minor ticks on by
default.  

Tick formatting
---------------

Tick formatting is controlled by classes derived from Formatter.  The
formatter operates on a single tick value and returns a string to the
axis.

  * NullFormatter      - no labels on the ticks

  * FixedFormatter     - set the strings manually for the labels

  * FuncFormatter      - user defined function sets the labels

  * FormatStrFormatter - use a sprintf format string

  * IndexFormatter     - cycle through fixed strings by tick position
  
  * ScalarFormatter    - default formatter for scalars; autopick the fmt string

  * LogFormatter       - formatter for log axes


You can derive your own formatter from the Formatter base class by
simply overriding the __call__ method.  The formatter class has access
to the axis view and data limits.

To control the major and minor tick label formats, use one of the
following methods::

  ax.xaxis.set_major_formatter( xmajorFormatter )
  ax.xaxis.set_minor_formatter( xminorFormatter )
  ax.yaxis.set_major_formatter( ymajorFormatter )
  ax.yaxis.set_minor_formatter( yminorFormatter )

See examples/major_minor_demo1.py for an example of setting major an
minor ticks.  See the matplotlib.dates module for more information and
examples of using date locators and formatters.

DEVELOPERS NOTE

If you are implementing your own class or modifying one of these, it
is critical that you use viewlim and dataInterval READ ONLY MODE so
multiple axes can share the same locator w/o side effects!


"""


from __future__ import division
import sys, os, re, time, math
from mlab import linspace
from matplotlib import verbose
from numerix import arange, array, asarray, ones, zeros, \
     nonzero, take, Float, log, logical_and

class TickHelper:

    viewInterval = None
    dataInterval = None

    def verify_intervals(self):
        if self.dataInterval is None:
            raise RuntimeError("You must set the data interval to use this function")

        if self.viewInterval is None:
            raise RuntimeError("You must set the view interval to use this function")

    
    def set_view_interval(self, interval):
        self.viewInterval = interval

    def set_data_interval(self, interval):
        self.dataInterval = interval

class Formatter(TickHelper):
    """
    Convert the tick location to a string
    """

    # some classes want to see all the locs to help format
    # individual ones    
    locs = None 
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'
        raise NotImplementedError('Derived must overide')

    def set_locs(self, locs):
        self.locs = locs

class NullFormatter(Formatter):
    'Always return the empty string'
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        return ''

class FixedFormatter(Formatter):
    'Return fixed strings for tick labels'
    def __init__(self, seq):
        """
        seq is a sequence of strings.  For positions i<len(seq) return
        seq[i] regardless of x.  Otherwise return ''
        """
        self.seq = seq
        
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        if pos>=len(self.seq): return ''
        else: return self.seq[pos]

class FuncFormatter(Formatter):
    """
    User defined function for formatting
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'                
        return self.func(x, pos)


class FormatStrFormatter(Formatter):
    """
    Use a format string to format the tick
    """
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'
        return self.fmt % x




class ScalarFormatter(Formatter):
    """
    Tick location is a plain old number.  If viewInterval is set, the
    formatter will use %d, %1.#f or %1.ef as appropriate.  If it is
    not set, the formatter will do str conversion
    """

    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        self.verify_intervals()
        d = abs(self.viewInterval.span())

        return self.pprint_val(x,d)

    def pprint_val(self, x, d):
        #if the number is not too big and it's an int, format it as an
        #int
        if abs(x)<1e4 and x==int(x): return '%d' % x
 
        if d < 1e-2: fmt = '%1.3e'
        elif d < 1e-1: fmt = '%1.3f'
        elif d > 1e5: fmt = '%1.1e'
        elif d > 10 : fmt = '%1.1f'
        elif d > 1 : fmt = '%1.2f'
        else: fmt = '%1.3f'
        s =  fmt % x

        tup = s.split('e')
        if len(tup)==2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            s = '%se%s%s' %(mantissa, sign, exponent)
        else:
            s = s.rstrip('0').rstrip('.')
        return s

class LogFormatter(ScalarFormatter):
    """
    Format values for log axis; 

    if attribute decadeOnly is True, only the decades will be labelled.
    """
    def __init__(self, base=10.0, labelOnlyBase = True):
        """
        base is used to locate the decade tick, 
        which will be the only one to be labeled if labelOnlyBase
        is False
        """
        self.base = base+0.0
        self.labelOnlyBase=labelOnlyBase
        self.decadeOnly = True

    def base(self,base):
        'change the base for labeling - warning: should always match the base used for LogLocator' 
        self.base=base
        
    def label_minor(self,labelOnlyBase):
        'switch on/off minor ticks labeling' 
        self.labelOnlyBase=labelOnlyBase

        
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        self.verify_intervals()
        d = abs(self.viewInterval.span())
        b=self.base
        # only label the decades
        fx = math.log(x)/math.log(b)
        isDecade = self.is_decade(fx)
        if not isDecade and self.labelOnlyBase: s = ''
        elif x>10000: s= '%1.0e'%x
        elif x<1: s =  '%1.0e'%x
        else        : s =  self.pprint_val(x,d)
        return s

    def is_decade(self, x):
        n = self.nearest_long(x)
        return abs(x-n)<1e-10
    
    def nearest_long(self, x):
        if x==0: return 0L
        elif x>0: return long(x+0.5)
        else: return long(x-0.5)
        
class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis; using exponent = log_base(value)
    """
        
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        self.verify_intervals()
        d = abs(self.viewInterval.span())
        b=self.base
        # only label the decades
        fx = math.log(x)/math.log(b)
        isDecade = self.is_decade(fx)
        if not isDecade and self.labelOnlyBase: s = ''
        #if 0: pass
        elif fx>10000: s= '%1.0e'%fx
        #elif x<1: s = '$10^{%d}$'%fx
        #elif x<1: s =  '10^%d'%fx
        elif fx<1: s =  '%1.0e'%fx
        else        : s =  self.pprint_val(fx,d)
        return s


class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis; using exponent = log_base(value)
    """
        
    def __call__(self, x, pos=0):
        'Return the format for tick val x at position pos'        
        self.verify_intervals()

        b = self.base
        # only label the decades
        fx = math.log(x)/math.log(b)
        isDecade = self.is_decade(fx)


        if not isDecade and self.labelOnlyBase: s = ''
        elif not isDecade: s = '$%d^{%.2f}$'% (b, fx)
        else: s = '$%d^{%d}$'% (b, self.nearest_long(fx))

        return s




class Locator(TickHelper):
    """
    Determine the tick locations
    """

    def __call__(self):
        'Return the locations of the ticks'
        raise NotImplementedError('Derived must override')

    def autoscale(self):
        'autoscale the view limits'
        self.verify_intervals()
        return  self.nonsingular(*self.dataInterval.get_bounds())

    def pan(self, numsteps):
        'Pan numticks (can be positive or negative)'
        ticks = self()
        numticks = len(ticks)

        if numticks>2:
            step = numsteps*abs(ticks[0]-ticks[1])
        else:            
            step = numsteps*self.viewInterval.span()/6

        self.viewInterval.shift(step)

    def zoom(self, direction):
        "Zoom in/out on axis; if direction is >0 zoom in, else zoom out"
        vmin, vmax = self.viewInterval.get_bounds()
        interval = self.viewInterval.span()
        step = 0.1*interval*direction
        self.viewInterval.set_bounds(vmin + step, vmax - step)

    def nonsingular(self, vmin, vmax):
        if vmin==vmax:
            if vmin==0.0:
                vmin -= 1
                vmax += 1
            else:
                vmin -= 0.001*abs(vmin)
                vmax += 0.001*abs(vmax)
        return vmin, vmax

    def refresh(self):
        'refresh internal information based on current lim'
        pass


class IndexLocator(Locator):
    """
    Place a tick on every multiple of some base number of points
    plotted, eg on every 5th point.  It is assumed that you are doing
    index plotting; ie the axis is 0, len(data).  This is mainly
    useful for x ticks.
    """
    def __init__(self, base, offset):
        'place ticks on the i-th data points where (i-offset)%base==0'
        self.base = base
        self.offset = offset

    def __call__(self):
        'Return the locations of the ticks'
        dmin, dmax = self.dataInterval.get_bounds()
        return arange(dmin + self.offset, dmax+1, self.base)


class FixedLocator(Locator):
    """
    Tick locations are fixed
    """

    def __init__(self, locs):
        self.locs = locs

    def __call__(self):
        'Return the locations of the ticks'
        return self.locs


class NullLocator(Locator):
    """
    No ticks
    """

    def __call__(self):
        'Return the locations of the ticks'
        return []

class LinearLocator(Locator):
    """
    Determine the tick locations

    The first time this function is called it will try and set the
    number of ticks to make a nice tick partitioning.  Thereafter the
    number of ticks will be fixed so that interactive navigation will
    be nice
    """
    

    def __init__(self, numticks = None, presets=None):
        """
        Use presets to set locs based on lom.  A dict mapping vmin, vmax->locs
        """
        self.numticks = numticks
        if presets is None:
            self.presets = {}            
        else:
            self.presets = presets
    def __call__(self):
        'Return the locations of the ticks'

        self.verify_intervals()
        vmin, vmax = self.viewInterval.get_bounds()
        if vmax<vmin:
            vmin, vmax = vmax, vmin

        if self.presets.has_key((vmin, vmax)):
            return self.presets[(vmin, vmax)]

        if self.numticks is None:
            self._set_numticks()
            


        if self.numticks==0: return []
        ticklocs = linspace(vmin, vmax, self.numticks)

        return ticklocs


    def _set_numticks(self):
        self.numticks = 11  # todo; be smart here; this is just for dev

    def autoscale(self):
        'Try to choose the view limits intelligently'
        self.verify_intervals()        
        vmin, vmax = self.dataInterval.get_bounds()

        if vmax<vmin:
            vmin, vmax = vmax, vmin

        if vmin==vmax:
            vmin-=1
            vmax+=1

        exponent, remainder = divmod(math.log10(vmax - vmin), 1)

        if remainder < 0.5:
            exponent -= 1
        scale = 10**(-exponent)
        vmin = math.floor(scale*vmin)/scale
        vmax = math.ceil(scale*vmax)/scale

        return self.nonsingular(vmin, vmax)


def closeto(x,y):
    if abs(x-y)<1e-10: return True
    else: return False

class Base:
    'this solution has some hacks to deal with floating point inaccuracies'
    def __init__(self, base):
        assert(base>0)
        self.base = base

    def lt(self, x):
        'return the largest multiple of base < x'
        d,m = divmod(x, self.base)
        if m==0: return (d-1)*self.base
        else: return d*self.base

    def le(self, x):
        'return the largest multiple of base <= x'
        d,m = divmod(x, self.base)
        if closeto(m, self.base):
            #looks like floating point error
            return (d+1)*self.base
        else:
            return d*self.base

    def gt(self, x):
        'return the largest multiple of base > x'
        d,m = divmod(x, self.base)
        if closeto(m, self.base):
            #looks like floating point error
            return (d+2)*self.base
        else:
            return (d+1)*self.base



    def ge(self, x):
        'return the largest multiple of base >= x'
        d,m = divmod(x, self.base)
        if m==0: return x
        return (d+1)*self.base

    def get_base(self):
        return self.base
    
class MultipleLocator(Locator):
    """
    Set a tick on every integer that is multiple of base in the
    viewInterval
    """

    def __init__(self, base=1.0):
        self.base = Base(base)
        
    def __call__(self):
        'Return the locations of the ticks'

        self.verify_intervals()
        
        vmin, vmax = self.viewInterval.get_bounds()
        if vmax<vmin:
            vmin, vmax = vmax, vmin
        vmin = self.base.ge(vmin)
        locs =  arange(vmin, vmax+0.001*self.base.get_base(), self.base.get_base())

        return locs

    def autoscale(self):
        """
        Set the view limits to the nearest multiples of base that
        contain the data
        """

        self.verify_intervals()
        dmin, dmax = self.dataInterval.get_bounds()

        vmin = self.base.le(dmin)
        vmax = self.base.ge(dmax)
        if vmin==vmax:
            vmin -=1
            vmax +=1

        
        return self.nonsingular(vmin, vmax)

def decade_down(x, base=10):
    'floor x to the nearest lower decade'

    lx = math.floor(math.log(x)/math.log(base))
    return base**lx

def decade_up(x, base=10):
    'ceil x to the nearest higher decade'
    lx = math.ceil(math.log(x)/math.log(base))
    return base**lx

def is_decade(x,base=10):
    lx = math.log(x)/math.log(base)
    return lx==int(lx)

class LogLocator(Locator):
    """
    Determine the tick locations for log axes
    """

    def __init__(self, base=10.0, subs=[1.0]):
        """
        place ticks on the location= base**i*subs[j]
        """
        self.base = base+0.0
        self.subs = array(subs)+0.0
        
    def base(self,base):
        """
        set the base of the log scaling (major tick every base**i, i interger)
        """
        self.base=base+0.0
        
    def subs(self,subs):
        """
        set the minor ticks the log scaling every base**i*subs[j] 
        """
        self.subs = array(subs)+0.0
    
    def __call__(self):
        'Return the locations of the ticks'
        self.verify_intervals()
        b=self.base
        subs=self.subs
        vmin, vmax = self.viewInterval.get_bounds()
        vmin = math.log(vmin)/math.log(b)
        vmax = math.log(vmax)/math.log(b)

        if vmax<vmin:
            vmin, vmax = vmax, vmin
        ticklocs = []
        for decadeStart in b**arange(math.floor(vmin),math.ceil(vmax)):
            ticklocs.extend( subs*decadeStart )

        if(len(subs) and subs[0]==1.0):
            ticklocs.append(b**math.ceil(vmax))

            
        ticklocs = array(ticklocs)
        ind = nonzero(logical_and(ticklocs>=b**vmin ,
                                  ticklocs<=b**vmax))

        
        ticklocs = take(ticklocs,ind)
        return ticklocs



    def autoscale(self):
        'Try to choose the view limits intelligently'
        self.verify_intervals()
        
        vmin, vmax = self.dataInterval.get_bounds()
        if vmax<vmin:
            vmin, vmax = vmax, vmin

        minpos = self.dataInterval.minpos()
        if minpos<0:
            raise RuntimeError('No positive data to plot')
        if vmin<0:
            vmin = minpos
         
        if not is_decade(vmin,self.base): vmin = decade_down(vmin,self.base)
        if not is_decade(vmax,self.base): vmax = decade_up(vmax,self.base)
        if vmin==vmax:
            vmin = decade_down(vmin,self.base)
            vmax = decade_up(vmax,self.base)
            
        return self.nonsingular(vmin, vmax)

class AutoLocator(Locator):
    """
    On autoscale this class picks the best MultipleLocator to set the
    view limits and the tick locs.  

    """
    def __init__(self):
        self._locator = LinearLocator()

    def __call__(self):
        'Return the locations of the ticks'
        self.refresh()
        return self._locator()

    def refresh(self):
        'refresh internal information based on current lim'
        d = self.viewInterval.span()
        self._locator = self.get_locator(d)

    def autoscale(self):
        'Try to choose the view limits intelligently'

        self.verify_intervals()
        d = abs(self.dataInterval.span())
        self._locator = self.get_locator(d)
        return self._locator.autoscale()

    def get_locator(self, d):
        'pick the best locator based on a distance'
        d = abs(d)
        if d<=0:
            locator = MultipleLocator(0.2)
        else:

            try: ld = math.log10(d)
            except OverflowError:
                verbose.report_error('AutoLocator illegal dataInterval range %s; returning NullLocator'%d)
                return NullLocator()

            fld = math.floor(ld)
            base = 10**fld

            #if ld==fld:  base = 10**(fld-1)
            #else:        base = 10**fld

            if   d >= 5*base : ticksize = base
            elif d >= 2*base : ticksize = base/2.0
            else             : ticksize = base/5.0
            #print 'base, ticksize, d', base, ticksize, d, self.viewInterval

            #print self.dataInterval, d, ticksize
            locator = MultipleLocator(ticksize)

        locator.set_view_interval(self.viewInterval)
        locator.set_data_interval(self.dataInterval)
        return locator



__all__ = ('TickHelper', 'Formatter', 'FixedFormatter',
           'NullFormatter', 'FuncFormatter', 'FormatStrFormatter',
           'ScalarFormatter', 'LogFormatter', 'LogFormatterExponent',
           'LogFormatterMathtext', 'Locator', 'IndexLocator',
           'FixedLocator', 'NullLocator', 'LinearLocator',
           'LogLocator', 'AutoLocator', 'MultipleLocator', )
