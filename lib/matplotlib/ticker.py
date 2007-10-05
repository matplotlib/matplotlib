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
limits, and the choosing of tick locations.  A useful semi-automatic
tick locator is MultipleLocator.  You initialize this with a base, eg
10, and it picks axis limits and ticks that are multiples of your
base.

The Locator subclasses defined here are

  * NullLocator     - No ticks

  * FixedLocator    - Tick locations are fixed

  * IndexLocator    - locator for index plots (eg where x = range(len(y))

  * LinearLocator   - evenly spaced ticks from min to max

  * LogLocator      - logarithmically ticks from min to max

  * MultipleLocator - ticks and range are a multiple of base;
                      either integer or float

  * OldAutoLocator  - choose a MultipleLocator and dyamically reassign
                      it for intelligent ticking during navigation

  * MaxNLocator     - finds up to a max number of ticks at nice
                      locations

  * AutoLocator     - MaxNLocator with simple defaults. This is the
                      default tick locator for most plotting.

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
import sys, os, re, time, math, warnings
import numpy as npy
import matplotlib as mpl
from matplotlib import verbose, rcParams
from matplotlib import cbook
from matplotlib import transforms as mtrans





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

    def set_bounds(self, vmin, vmax):
        '''
        Set dataInterval and viewInterval from numeric vmin, vmax.

        This is for stand-alone use of Formatters and/or
        Locators that require these intervals; that is, for
        cases where the Intervals do not need to be updated
        automatically.
        '''
        self.dataInterval = mtrans.Interval(mtrans.Value(vmin), mtrans.Value(vmax))
        self.viewInterval = mtrans.Interval(mtrans.Value(vmin), mtrans.Value(vmax))

class Formatter(TickHelper):
    """
    Convert the tick location to a string
    """

    # some classes want to see all the locs to help format
    # individual ones
    locs = []
    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos; pos=None indicated unspecified'
        raise NotImplementedError('Derived must overide')

    def format_data(self,value):
        return self.__call__(value)

    def format_data_short(self,value):
        'return a short string version'
        return self.format_data(value)

    def get_offset(self):
        return ''

    def set_locs(self, locs):
        self.locs = locs

class NullFormatter(Formatter):
    'Always return the empty string'
    def __call__(self, x, pos=None):
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
        self.offset_string = ''

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        if pos is None or pos>=len(self.seq): return ''
        else: return self.seq[pos]

    def get_offset(self):
        return self.offset_string

    def set_offset_string(self, ofs):
        self.offset_string = ofs

class FuncFormatter(Formatter):
    """
    User defined function for formatting
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        return self.func(x, pos)


class FormatStrFormatter(Formatter):
    """
    Use a format string to format the tick
    """
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        return self.fmt % x

class OldScalarFormatter(Formatter):
    """
    Tick location is a plain old number.  If viewInterval is set, the
    formatter will use %d, %1.#f or %1.ef as appropriate.  If it is
    not set, the formatter will do str conversion
    """

    def __call__(self, x, pos=None):
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
        #print d, x, fmt, s
        tup = s.split('e')
        if len(tup)==2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            s = '%se%s%s' %(mantissa, sign, exponent)
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class ScalarFormatter(Formatter):
    """
    Tick location is a plain old number.  If useOffset==True and the data range
    is much smaller than the data average, then an offset will be determined
    such that the tick labels are meaningful. Scientific notation is used for
    data < 1e-3 or data >= 1e4.
    """
    def __init__(self, useOffset=True, useMathText=False):
        # useOffset allows plotting small data ranges with large offsets:
        # for example: [1+1e-9,1+2e-9,1+3e-9]
        # useMathText will render the offset and scientific notation in mathtext
        self._useOffset = useOffset
        self._usetex = rcParams['text.usetex']
        self._useMathText = useMathText
        self.offset = 0
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = rcParams['axes.formatter.limits']

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        if len(self.locs)==0:
            return ''
        else:
            return self.pprint_val(x)

    def set_scientific(self, b):
        '''True or False to turn scientific notation on or off
        see also set_powerlimits()
        '''
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        '''
        Sets size thresholds for scientific notation.

        e.g. xaxis.set_powerlimits((-3, 4)) sets the pre-2007 default in
        which scientific notation is used for numbers less than
        1e-3 or greater than 1e4.
        See also set_scientific().
        '''
        assert len(lims) == 2, "argument must be a sequence of length 2"
        self._powerlimits = lims

    def format_data_short(self,value):
        'return a short formatted string representation of a number'
        return '%1.3g'%value

    def format_data(self,value):
        'return a formatted string representation of a number'
        return self._formatSciNotation('%1.10e'% value)

    def get_offset(self):
        """Return scientific notation, plus offset"""
        if len(self.locs)==0: return ''
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''
            sciNotStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0: offsetStr = '+' + offsetStr
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = r'{\times}'+self.format_data(10**self.orderOfMagnitude)
                else:
                    sciNotStr = u'\xd7'+'1e%d'% self.orderOfMagnitude
            if self._useMathText:
                return ''.join(('$\mathdefault{',sciNotStr,offsetStr,'}$'))
            elif self._usetex:
                return ''.join(('$',sciNotStr,offsetStr,'$'))
            else:
                return ''.join((sciNotStr,offsetStr))
        else: return ''

    def set_locs(self, locs):
        'set the locations of the ticks'
        self.locs = locs
        if len(self.locs) > 0:
            self.verify_intervals()
            d = abs(self.viewInterval.span())
            if self._useOffset: self._set_offset(d)
            self._set_orderOfMagnitude(d)
            self._set_format()

    def _set_offset(self, range):
        # offset of 20,001 is 20,000, for example
        locs = self.locs

        if locs is None or not len(locs) or range == 0:
            self.offset = 0
            return
        ave_loc = npy.mean(locs)
        if ave_loc: # dont want to take log10(0)
            ave_oom = math.floor(math.log10(npy.mean(npy.absolute(locs))))
            range_oom = math.floor(math.log10(range))

            if npy.absolute(ave_oom-range_oom) >= 3: # four sig-figs
                if ave_loc < 0:
                    self.offset = math.ceil(npy.max(locs)/10**range_oom)*10**range_oom
                else:
                    self.offset = math.floor(npy.min(locs)/10**(range_oom))*10**(range_oom)
            else: self.offset = 0

    def _set_orderOfMagnitude(self,range):
        # if scientific notation is to be used, find the appropriate exponent
        # if using an numerical offset, find the exponent after applying the offset
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        locs = npy.absolute(self.locs)
        if self.offset: oom = math.floor(math.log10(range))
        else:
            if locs[0] > locs[-1]: val = locs[0]
            else: val = locs[-1]
            if val == 0: oom = 0
            else: oom = math.floor(math.log10(val))
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self):
        # set the format string to format all the ticklabels
        # The floating point black magic (adding 1e-15 and formatting
        # to 8 digits) may warrant review and cleanup.
        locs = (npy.asarray(self.locs)-self.offset) / 10**self.orderOfMagnitude+1e-15
        sigfigs = [len(str('%1.8f'% loc).split('.')[1].rstrip('0')) \
                   for loc in locs]
        sigfigs.sort()
        self.format = '%1.' + str(sigfigs[-1]) + 'f'
        if self._usetex:
            self.format = '$%s$' % self.format
        elif self._useMathText:
            self.format = '$\mathdefault{%s}$' % self.format

    def pprint_val(self, x):
        xp = (x-self.offset)/10**self.orderOfMagnitude
        if npy.absolute(xp) < 1e-8: xp = 0
        return self.format % xp

    def _formatSciNotation(self, s):
        # transform 1e+004 into 1e4, for example
        tup = s.split('e')
        try:
            significand = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            if self._useMathText or self._usetex:
                if significand == '1':
                    # reformat 1x10^y as 10^y
                    significand = ''
                if exponent:
                    exponent = '10^{%s%s}'%(sign, exponent)
                if significand and exponent:
                    return r'%s{\times}%s'%(significand, exponent)
                else:
                    return r'%s%s'%(significand, exponent)
            else:
                return ('%se%s%s' %(significand, sign, exponent)).rstrip('e')
        except IndexError, msg:
            return s


class LogFormatter(Formatter):
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
        self._base = base+0.0
        self.labelOnlyBase=labelOnlyBase
        self.decadeOnly = True

    def base(self,base):
        'change the base for labeling - warning: should always match the base used for LogLocator'
        self._base=base

    def label_minor(self,labelOnlyBase):
        'switch on/off minor ticks labeling'
        self.labelOnlyBase=labelOnlyBase


    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        self.verify_intervals()
        d = abs(self.viewInterval.span())
        b=self._base
        # only label the decades
        fx = math.log(x)/math.log(b)
        isDecade = self.is_decade(fx)
        if not isDecade and self.labelOnlyBase: s = ''
        elif x>10000: s= '%1.0e'%x
        elif x<1: s =  '%1.0e'%x
        else        : s =  self.pprint_val(x,d)
        return s

    def format_data(self,value):
        self.labelOnlyBase = False
        value = cbook.strip_math(self.__call__(value))
        self.labelOnlyBase = True
        return value

    def is_decade(self, x):
        n = self.nearest_long(x)
        return abs(x-n)<1e-10

    def nearest_long(self, x):
        if x==0: return 0L
        elif x>0: return long(x+0.5)
        else: return long(x-0.5)

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
        #print d, x, fmt, s
        tup = s.split('e')
        if len(tup)==2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            s = '%se%s%s' %(mantissa, sign, exponent)
        else:
            s = s.rstrip('0').rstrip('.')
        return s

class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis; using exponent = log_base(value)
    """

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        self.verify_intervals()
        d = abs(self.viewInterval.span())
        b=self._base
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

    def __call__(self, x, pos=None):
        'Return the format for tick val x at position pos'
        self.verify_intervals()

        b = self._base
        # only label the decades
        fx = math.log(x)/math.log(b)
        isDecade = self.is_decade(fx)

        usetex = rcParams['text.usetex']

        if not isDecade and self.labelOnlyBase: s = ''
        elif not isDecade:
            if usetex:
                s = r'$%d^{%.2f}$'% (b, fx)
            else:
                s = '$\mathdefault{%d^{%.2f}}$'% (b, fx)
        else:
            if usetex:
                s = r'$%d^{%d}$'% (b, self.nearest_long(fx))
            else:
                s = r'$\mathdefault{%d^{%d}}$'% (b, self.nearest_long(fx))

        return s




class Locator(TickHelper):
    """
    Determine the tick locations;

    Note, you should not use the same locator between different Axis
    because the locator stores references to the Axis data and view
    limits
    """

    def __call__(self):
        'Return the locations of the ticks'
        raise NotImplementedError('Derived must override')

    def autoscale(self):
        'autoscale the view limits'
        self.verify_intervals()
        return mtrans.nonsingular(*self.dataInterval.get_bounds())

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
        self._base = base
        self.offset = offset

    def __call__(self):
        'Return the locations of the ticks'
        dmin, dmax = self.dataInterval.get_bounds()
        return npy.arange(dmin + self.offset, dmax+1, self._base)


class FixedLocator(Locator):
    """
    Tick locations are fixed.  If nbins is not None,
    the array of possible positions will be subsampled to
    keep the number of ticks <= nbins +1.
    """

    def __init__(self, locs, nbins=None):
        self.locs = locs
        self.nbins = nbins
        if self.nbins is not None:
            self.nbins = max(self.nbins, 2)

    def __call__(self):
        'Return the locations of the ticks'
        if self.nbins is None:
            return self.locs
        step = max(int(0.99 + len(self.locs) / float(self.nbins)), 1)
        return self.locs[::step]




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

    The first time this function is called it will try to set the
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
        ticklocs = npy.linspace(vmin, vmax, self.numticks)

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

        return mtrans.nonsingular(vmin, vmax)


def closeto(x,y):
    if abs(x-y)<1e-10: return True
    else: return False

class Base:
    'this solution has some hacks to deal with floating point inaccuracies'
    def __init__(self, base):
        assert(base>0)
        self._base = base

    def lt(self, x):
        'return the largest multiple of base < x'
        d,m = divmod(x, self._base)
        if closeto(m,0) and not closeto(m/self._base,1):
            return (d-1)*self._base
        return d*self._base

    def le(self, x):
        'return the largest multiple of base <= x'
        d,m = divmod(x, self._base)
        if closeto(m/self._base,1): # was closeto(m, self._base)
            #looks like floating point error
            return (d+1)*self._base
        return d*self._base

    def gt(self, x):
        'return the smallest multiple of base > x'
        d,m = divmod(x, self._base)
        if closeto(m/self._base,1):
            #looks like floating point error
            return (d+2)*self._base
        return (d+1)*self._base

    def ge(self, x):
        'return the smallest multiple of base >= x'
        d,m = divmod(x, self._base)
        if closeto(m,0) and not closeto(m/self._base,1):
            return d*self._base
        return (d+1)*self._base

    def get_base(self):
        return self._base

class MultipleLocator(Locator):
    """
    Set a tick on every integer that is multiple of base in the
    viewInterval
    """

    def __init__(self, base=1.0):
        self._base = Base(base)

    def __call__(self):
        'Return the locations of the ticks'

        self.verify_intervals()

        vmin, vmax = self.viewInterval.get_bounds()
        if vmax<vmin:
            vmin, vmax = vmax, vmin
        vmin = self._base.ge(vmin)
        base = self._base.get_base()
        n = (vmax - vmin + 0.001*base)//base
        locs = vmin + npy.arange(n+1) * base
        return locs

    def autoscale(self):
        """
        Set the view limits to the nearest multiples of base that
        contain the data
        """

        self.verify_intervals()
        dmin, dmax = self.dataInterval.get_bounds()

        vmin = self._base.le(dmin)
        vmax = self._base.ge(dmax)
        if vmin==vmax:
            vmin -=1
            vmax +=1

        return mtrans.nonsingular(vmin, vmax)

def scale_range(vmin, vmax, n = 1, threshold=100):
    dv = abs(vmax - vmin)
    maxabsv = max(abs(vmin), abs(vmax))
    if maxabsv == 0 or dv/maxabsv < 1e-12:
        return 1.0, 0.0
    meanv = 0.5*(vmax+vmin)
    if abs(meanv)/dv < threshold:
        offset = 0
    elif meanv > 0:
        ex = divmod(math.log10(meanv), 1)[0]
        offset = 10**ex
    else:
        ex = divmod(math.log10(-meanv), 1)[0]
        offset = -10**ex
    ex = divmod(math.log10(dv/n), 1)[0]
    scale = 10**ex
    return scale, offset



class MaxNLocator(Locator):
    """
    Select no more than N intervals at nice locations.
    """

    def __init__(self, nbins = 10, steps = None, trim = True, integer=False):
        self._nbins = int(nbins)
        self._trim = trim
        self._integer = integer
        if steps is None:
            self._steps = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
        else:
            if int(steps[-1]) != 10:
                steps = list(steps)
                steps.append(10)
            self._steps = steps
        if integer:
            self._steps = [n for n in self._steps if divmod(n,1)[1] < 0.001]

    def bin_boundaries(self, vmin, vmax):
        nbins = self._nbins
        scale, offset = scale_range(vmin, vmax, nbins)
        if self._integer:
            scale = max(1, scale)
        vmin -= offset
        vmax -= offset
        raw_step = (vmax-vmin)/nbins
        scaled_raw_step = raw_step/scale

        for step in self._steps:
            if step < scaled_raw_step:
                continue
            step *= scale
            best_vmin = step*divmod(vmin, step)[0]
            best_vmax = best_vmin + step*nbins
            if (best_vmax >= vmax):
                break
        if self._trim:
            extra_bins = int(divmod((best_vmax - vmax), step)[0])
            nbins -= extra_bins
        return (npy.arange(nbins+1) * step + best_vmin + offset)


    def __call__(self):
        self.verify_intervals()
        vmin, vmax = self.viewInterval.get_bounds()
        vmin, vmax = mtrans.nonsingular(vmin, vmax, expander = 0.05)
        return self.bin_boundaries(vmin, vmax)

    def autoscale(self):
        self.verify_intervals()
        dmin, dmax = self.dataInterval.get_bounds()
        dmin, dmax = mtrans.nonsingular(dmin, dmax, expander = 0.05)
        return npy.take(self.bin_boundaries(dmin, dmax), [0,-1])


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
        self.base(base)
        self.subs(subs)
        self.numticks = 15

    def base(self,base):
        """
        set the base of the log scaling (major tick every base**i, i interger)
        """
        self._base=base+0.0

    def subs(self,subs):
        """
        set the minor ticks the log scaling every base**i*subs[j]
        """
        if subs is None:
            self._subs = None  # autosub
        else:
            self._subs = npy.asarray(subs)+0.0

    def _set_numticks(self):
        self.numticks = 15  # todo; be smart here; this is just for dev

    def __call__(self):
        'Return the locations of the ticks'
        self.verify_intervals()
        b=self._base

        vmin, vmax = self.viewInterval.get_bounds()
        vmin = math.log(vmin)/math.log(b)
        vmax = math.log(vmax)/math.log(b)

        if vmax<vmin:
            vmin, vmax = vmax, vmin
        ticklocs = []

        numdec = math.floor(vmax)-math.ceil(vmin)

        if self._subs is None: # autosub
            if numdec>10: subs = npy.array([1.0])
            elif numdec>6: subs = npy.arange(2.0, b, 2.0)
            else: subs = npy.arange(2.0, b)
        else:
            subs = self._subs

        stride = 1
        while numdec/stride+1 > self.numticks:
            stride += 1

        for decadeStart in b**npy.arange(math.floor(vmin),
                                         math.ceil(vmax)+stride, stride):
            ticklocs.extend( subs*decadeStart )

        return npy.array(ticklocs)

    def autoscale(self):
        'Try to choose the view limits intelligently'
        self.verify_intervals()

        vmin, vmax = self.dataInterval.get_bounds()
        if vmax<vmin:
            vmin, vmax = vmax, vmin

        minpos = self.dataInterval.minpos()

        if minpos<=0:
            raise RuntimeError('No positive data to plot')
        if vmin<=0:
            vmin = minpos
        if not is_decade(vmin,self._base): vmin = decade_down(vmin,self._base)
        if not is_decade(vmax,self._base): vmax = decade_up(vmax,self._base)
        if vmin==vmax:
            vmin = decade_down(vmin,self._base)
            vmax = decade_up(vmax,self._base)
        return mtrans.nonsingular(vmin, vmax)

class AutoLocator(MaxNLocator):
    def __init__(self):
        MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])

class OldAutoLocator(Locator):
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
                raise RuntimeError('AutoLocator illegal dataInterval range')


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
           'LogLocator', 'AutoLocator', 'MultipleLocator',
           'MaxNLocator', )
