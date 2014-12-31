"""
Tick locating and formatting
============================

This module contains classes to support completely configurable tick locating
and formatting.  Although the locators know nothing about major or minor
ticks, they are used by the Axis class to support major and minor tick
locating and formatting.  Generic tick locators and formatters are provided,
as well as domain specific custom ones..


Default Formatter
-----------------

The default formatter identifies when the x-data being
plotted is a small range on top of a large off set.  To
reduce the chances that the ticklabels overlap the ticks
are labeled as deltas from a fixed offset.  For example::

   ax.plot(np.arange(2000, 2010), range(10))

will have tick of 0-9 with an offset of +2e3.  If this
is not desired turn off the use of the offset on the default
formatter::


   ax.get_xaxis().get_major_formatter().set_useOffset(False)

set the rcParam ``axes.formatter.useoffset=False`` to turn it off
globally, or set a different formatter.

Tick locating
-------------

The Locator class is the base class for all tick locators.  The locators
handle autoscaling of the view limits based on the data limits, and the
choosing of tick locations.  A useful semi-automatic tick locator is
MultipleLocator.  You initialize this with a base, e.g., 10, and it picks axis
limits and ticks that are multiples of your base.

The Locator subclasses defined here are

:class:`NullLocator`
    No ticks

:class:`FixedLocator`
    Tick locations are fixed

:class:`IndexLocator`
    locator for index plots (e.g., where x = range(len(y)))

:class:`LinearLocator`
    evenly spaced ticks from min to max

:class:`LogLocator`
    logarithmically ticks from min to max

:class:`MultipleLocator`
    ticks and range are a multiple of base;
                  either integer or float
:class:`OldAutoLocator`
    choose a MultipleLocator and dyamically reassign it for
    intelligent ticking during navigation

:class:`MaxNLocator`
    finds up to a max number of ticks at nice locations

:class:`AutoLocator`
    :class:`MaxNLocator` with simple defaults. This is the default
    tick locator for most plotting.

:class:`AutoMinorLocator`
    locator for minor ticks when the axis is linear and the
    major ticks are uniformly spaced.  It subdivides the major
    tick interval into a specified number of minor intervals,
    defaulting to 4 or 5 depending on the major interval.


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

The default minor locator is the NullLocator, e.g., no minor ticks on by
default.

Tick formatting
---------------

Tick formatting is controlled by classes derived from Formatter.  The
formatter operates on a single tick value and returns a string to the
axis.

:class:`NullFormatter`
    no labels on the ticks

:class:`IndexFormatter`
    set the strings from a list of labels

:class:`FixedFormatter`
    set the strings manually for the labels

:class:`FuncFormatter`
    user defined function sets the labels

:class:`FormatStrFormatter`
    use a sprintf format string

:class:`ScalarFormatter`
    default formatter for scalars; autopick the fmt string

:class:`LogFormatter`
    formatter for log axes


You can derive your own formatter from the Formatter base class by
simply overriding the ``__call__`` method.  The formatter class has access
to the axis view and data limits.

To control the major and minor tick label formats, use one of the
following methods::

  ax.xaxis.set_major_formatter( xmajorFormatter )
  ax.xaxis.set_minor_formatter( xminorFormatter )
  ax.yaxis.set_major_formatter( ymajorFormatter )
  ax.yaxis.set_minor_formatter( yminorFormatter )

See :ref:`pylab_examples-major_minor_demo1` for an example of setting
major and minor ticks.  See the :mod:`matplotlib.dates` module for
more information and examples of using date locators and formatters.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
if six.PY3:
    long = int

import decimal
import locale
import math
import numpy as np
from matplotlib import rcParams
from matplotlib import cbook
from matplotlib import transforms as mtransforms


class _DummyAxis(object):
    def __init__(self, minpos=0):
        self.dataLim = mtransforms.Bbox.unit()
        self.viewLim = mtransforms.Bbox.unit()
        self._minpos = minpos

    def get_view_interval(self):
        return self.viewLim.intervalx

    def set_view_interval(self, vmin, vmax):
        self.viewLim.intervalx = vmin, vmax

    def get_minpos(self):
        return self._minpos

    def get_data_interval(self):
        return self.dataLim.intervalx

    def set_data_interval(self, vmin, vmax):
        self.dataLim.intervalx = vmin, vmax


class TickHelper(object):
    axis = None

    def set_axis(self, axis):
        self.axis = axis

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.axis = _DummyAxis(**kwargs)

    def set_view_interval(self, vmin, vmax):
        self.axis.set_view_interval(vmin, vmax)

    def set_data_interval(self, vmin, vmax):
        self.axis.set_data_interval(vmin, vmax)

    def set_bounds(self, vmin, vmax):
        self.set_view_interval(vmin, vmax)
        self.set_data_interval(vmin, vmax)


class Formatter(TickHelper):
    """
    Convert the tick location to a string
    """
    # some classes want to see all the locs to help format
    # individual ones
    locs = []

    def __call__(self, x, pos=None):
        """Return the format for tick val x at position pos; pos=None
           indicated unspecified"""
        raise NotImplementedError('Derived must override')

    def format_data(self, value):
        return self.__call__(value)

    def format_data_short(self, value):
        """return a short string version"""
        return self.format_data(value)

    def get_offset(self):
        return ''

    def set_locs(self, locs):
        self.locs = locs

    def fix_minus(self, s):
        """
        Some classes may want to replace a hyphen for minus with the
        proper unicode symbol (U+2212) for typographical correctness.
        The default is to not replace it.

        Note, if you use this method, e.g., in :meth:`format_data` or
        call, you probably don't want to use it for
        :meth:`format_data_short` since the toolbar uses this for
        interactive coord reporting and I doubt we can expect GUIs
        across platforms will handle the unicode correctly.  So for
        now the classes that override :meth:`fix_minus` should have an
        explicit :meth:`format_data_short` method
        """
        return s


class IndexFormatter(Formatter):
    """
    format the position x to the nearest i-th label where i=int(x+0.5)
    """
    def __init__(self, labels):
        self.labels = labels
        self.n = len(labels)

    def __call__(self, x, pos=None):
        """Return the format for tick val x at position pos; pos=None
        indicated unspecified"""
        i = int(x + 0.5)
        if i < 0:
            return ''
        elif i >= self.n:
            return ''
        else:
            return self.labels[i]


class NullFormatter(Formatter):
    'Always return the empty string'

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        return ''


class FixedFormatter(Formatter):
    'Return fixed strings for tick labels'
    def __init__(self, seq):
        """
        *seq* is a sequence of strings.  For positions ``i < len(seq)`` return
        *seq[i]* regardless of *x*.  Otherwise return ''
        """
        self.seq = seq
        self.offset_string = ''

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        if pos is None or pos >= len(self.seq):
            return ''
        else:
            return self.seq[pos]

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
        'Return the format for tick val *x* at position *pos*'
        return self.func(x, pos)


class FormatStrFormatter(Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick
    """
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        return self.fmt % x


class StrMethodFormatter(Formatter):
    """
    Use a new-style format string (as used by `str.format()`)
    to format the tick.  The field formatting must be labeled `x`.
    """
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        return self.fmt.format(x=x)


class OldScalarFormatter(Formatter):
    """
    Tick location is a plain old number.
    """

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        xmin, xmax = self.axis.get_view_interval()
        d = abs(xmax - xmin)

        return self.pprint_val(x, d)

    def pprint_val(self, x, d):
        #if the number is not too big and it's an int, format it as an
        #int
        if abs(x) < 1e4 and x == int(x):
            return '%d' % x

        if d < 1e-2:
            fmt = '%1.3e'
        elif d < 1e-1:
            fmt = '%1.3f'
        elif d > 1e5:
            fmt = '%1.1e'
        elif d > 10:
            fmt = '%1.1f'
        elif d > 1:
            fmt = '%1.2f'
        else:
            fmt = '%1.3f'
        s = fmt % x
        #print d, x, fmt, s
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            s = '%se%s%s' % (mantissa, sign, exponent)
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class ScalarFormatter(Formatter):
    """
    Tick location is a plain old number.  If useOffset==True and the data range
    is much smaller than the data average, then an offset will be determined
    such that the tick labels are meaningful. Scientific notation is used for
    data < 10^-n or data >= 10^m, where n and m are the power limits set using
    set_powerlimits((n,m)). The defaults for these are controlled by the
    axes.formatter.limits rc parameter.

    """

    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        # useOffset allows plotting small data ranges with large offsets: for
        # example: [1+1e-9,1+2e-9,1+3e-9] useMathText will render the offset
        # and scientific notation in mathtext

        if useOffset is None:
            useOffset = rcParams['axes.formatter.useoffset']
        self.set_useOffset(useOffset)
        self._usetex = rcParams['text.usetex']
        if useMathText is None:
            useMathText = rcParams['axes.formatter.use_mathtext']
        self._useMathText = useMathText
        self.orderOfMagnitude = 0
        self.format = ''
        self._scientific = True
        self._powerlimits = rcParams['axes.formatter.limits']
        if useLocale is None:
            useLocale = rcParams['axes.formatter.use_locale']
        self._useLocale = useLocale

    def get_useOffset(self):
        return self._useOffset

    def set_useOffset(self, val):
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val

    useOffset = property(fget=get_useOffset, fset=set_useOffset)

    def get_useLocale(self):
        return self._useLocale

    def set_useLocale(self, val):
        if val is None:
            self._useLocale = rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = val

    useLocale = property(fget=get_useLocale, fset=set_useLocale)

    def fix_minus(self, s):
        """use a unicode minus rather than hyphen"""
        if rcParams['text.usetex'] or not rcParams['axes.unicode_minus']:
            return s
        else:
            return s.replace('-', '\u2212')

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        if len(self.locs) == 0:
            return ''
        else:
            s = self.pprint_val(x)
            return self.fix_minus(s)

    def set_scientific(self, b):
        '''True or False to turn scientific notation on or off
        see also :meth:`set_powerlimits`
        '''
        self._scientific = bool(b)

    def set_powerlimits(self, lims):
        '''
        Sets size thresholds for scientific notation.

        e.g., ``formatter.set_powerlimits((-3, 4))`` sets the pre-2007 default
        in which scientific notation is used for numbers less than 1e-3 or
        greater than 1e4.
        See also :meth:`set_scientific`.
        '''
        assert len(lims) == 2, "argument must be a sequence of length 2"
        self._powerlimits = lims

    def format_data_short(self, value):
        """return a short formatted string representation of a number"""
        if self._useLocale:
            return locale.format_string('%-12g', (value,))
        else:
            return '%-12g' % value

    def format_data(self, value):
        'return a formatted string representation of a number'
        if self._useLocale:
            s = locale.format_string('%1.10e', (value,))
        else:
            s = '%1.10e' % value
        s = self._formatSciNotation(s)
        return self.fix_minus(s)

    def get_offset(self):
        """Return scientific notation, plus offset"""
        if len(self.locs) == 0:
            return ''
        s = ''
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''
            sciNotStr = ''
            if self.offset:
                offsetStr = self.format_data(self.offset)
                if self.offset > 0:
                    offsetStr = '+' + offsetStr
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = self.format_data(10 ** self.orderOfMagnitude)
                else:
                    sciNotStr = '1e%d' % self.orderOfMagnitude
            if self._useMathText:
                if sciNotStr != '':
                    sciNotStr = r'\times\mathdefault{%s}' % sciNotStr
                s = ''.join(('$', sciNotStr,
                             r'\mathdefault{', offsetStr, '}$'))
            elif self._usetex:
                if sciNotStr != '':
                    sciNotStr = r'\times%s' % sciNotStr
                s = ''.join(('$', sciNotStr, offsetStr, '$'))
            else:
                s = ''.join((sciNotStr, offsetStr))

        return self.fix_minus(s)

    def set_locs(self, locs):
        'set the locations of the ticks'
        self.locs = locs
        if len(self.locs) > 0:
            vmin, vmax = self.axis.get_view_interval()
            d = abs(vmax - vmin)
            if self._useOffset:
                self._set_offset(d)
            self._set_orderOfMagnitude(d)
            self._set_format(vmin, vmax)

    def _set_offset(self, range):
        # offset of 20,001 is 20,000, for example
        locs = self.locs

        if locs is None or not len(locs) or range == 0:
            self.offset = 0
            return
        ave_loc = np.mean(locs)
        if ave_loc:  # dont want to take log10(0)
            ave_oom = math.floor(math.log10(np.mean(np.absolute(locs))))
            range_oom = math.floor(math.log10(range))

            if np.absolute(ave_oom - range_oom) >= 3:  # four sig-figs
                p10 = 10 ** range_oom
                if ave_loc < 0:
                    self.offset = (math.ceil(np.max(locs) / p10) * p10)
                else:
                    self.offset = (math.floor(np.min(locs) / p10) * p10)
            else:
                self.offset = 0

    def _set_orderOfMagnitude(self, range):
        # if scientific notation is to be used, find the appropriate exponent
        # if using an numerical offset, find the exponent after applying the
        # offset
        if not self._scientific:
            self.orderOfMagnitude = 0
            return
        locs = np.absolute(self.locs)
        if self.offset:
            oom = math.floor(math.log10(range))
        else:
            if locs[0] > locs[-1]:
                val = locs[0]
            else:
                val = locs[-1]
            if val == 0:
                oom = 0
            else:
                oom = math.floor(math.log10(val))
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self, vmin, vmax):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = list(self.locs) + [vmin, vmax]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = '%1.' + str(sigfigs) + 'f'
        if self._usetex:
            self.format = '$%s$' % self.format
        elif self._useMathText:
            self.format = '$\mathdefault{%s}$' % self.format

    def pprint_val(self, x):
        xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
        if np.absolute(xp) < 1e-8:
            xp = 0
        if self._useLocale:
            return locale.format_string(self.format, (xp,))
        else:
            return self.format % xp

    def _formatSciNotation(self, s):
        # transform 1e+004 into 1e4, for example
        if self._useLocale:
            decimal_point = locale.localeconv()['decimal_point']
            positive_sign = locale.localeconv()['positive_sign']
        else:
            decimal_point = '.'
            positive_sign = '+'
        tup = s.split('e')
        try:
            significand = tup[0].rstrip('0').rstrip(decimal_point)
            sign = tup[1][0].replace(positive_sign, '')
            exponent = tup[1][1:].lstrip('0')
            if self._useMathText or self._usetex:
                if significand == '1' and exponent != '':
                    # reformat 1x10^y as 10^y
                    significand = ''
                if exponent:
                    exponent = '10^{%s%s}' % (sign, exponent)
                if significand and exponent:
                    return r'%s{\times}%s' % (significand, exponent)
                else:
                    return r'%s%s' % (significand, exponent)
            else:
                s = ('%se%s%s' % (significand, sign, exponent)).rstrip('e')
                return s
        except IndexError:
            return s


class LogFormatter(Formatter):
    """
    Format values for log axis;

    """
    def __init__(self, base=10.0, labelOnlyBase=True):
        """
        *base* is used to locate the decade tick,
        which will be the only one to be labeled if *labelOnlyBase*
        is ``False``
        """
        self._base = base + 0.0
        self.labelOnlyBase = labelOnlyBase

    def base(self, base):
        """change the *base* for labeling - warning: should always match the
           base used for :class:`LogLocator`"""
        self._base = base

    def label_minor(self, labelOnlyBase):
        'switch on/off minor ticks labeling'
        self.labelOnlyBase = labelOnlyBase

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        vmin, vmax = self.axis.get_view_interval()
        d = abs(vmax - vmin)
        b = self._base
        if x == 0.0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        isDecade = is_close_to_int(fx)
        if not isDecade and self.labelOnlyBase:
            s = ''
        elif x > 10000:
            s = '%1.0e' % x
        elif x < 1:
            s = '%1.0e' % x
        else:
            s = self.pprint_val(x, d)
        if sign == -1:
            s = '-%s' % s

        return self.fix_minus(s)

    def format_data(self, value):
        b = self.labelOnlyBase
        self.labelOnlyBase = False
        value = cbook.strip_math(self.__call__(value))
        self.labelOnlyBase = b
        return value

    def format_data_short(self, value):
        'return a short formatted string representation of a number'
        return '%-12g' % value

    def pprint_val(self, x, d):
        #if the number is not too big and it's an int, format it as an
        #int
        if abs(x) < 1e4 and x == int(x):
            return '%d' % x

        if d < 1e-2:
            fmt = '%1.3e'
        elif d < 1e-1:
            fmt = '%1.3f'
        elif d > 1e5:
            fmt = '%1.1e'
        elif d > 10:
            fmt = '%1.1f'
        elif d > 1:
            fmt = '%1.2f'
        else:
            fmt = '%1.3f'
        s = fmt % x
        #print d, x, fmt, s
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            sign = tup[1][0].replace('+', '')
            exponent = tup[1][1:].lstrip('0')
            s = '%se%s%s' % (mantissa, sign, exponent)
        else:
            s = s.rstrip('0').rstrip('.')
        return s


class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis; using ``exponent = log_base(value)``
    """

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        d = abs(vmax - vmin)
        b = self._base
        if x == 0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        isDecade = is_close_to_int(fx)
        if not isDecade and self.labelOnlyBase:
            s = ''
        elif abs(fx) > 10000:
            s = '%1.0g' % fx
        elif abs(fx) < 1:
            s = '%1.0g' % fx
        else:
            s = self.pprint_val(fx, d)
        if sign == -1:
            s = '-%s' % s

        return self.fix_minus(s)


class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis; using ``exponent = log_base(value)``
    """

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        b = self._base
        usetex = rcParams['text.usetex']

        # only label the decades
        if x == 0:
            if usetex:
                return '$0$'
            else:
                return '$\mathdefault{0}$'

        fx = math.log(abs(x)) / math.log(b)
        is_decade = is_close_to_int(fx)

        sign_string = '-' if x < 0 else ''

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        if not is_decade and self.labelOnlyBase:
            return ''
        elif not is_decade:
            if usetex:
                return (r'$%s%s^{%.2f}$') % \
                                            (sign_string, base, fx)
            else:
                return ('$\mathdefault{%s%s^{%.2f}}$') % \
                                            (sign_string, base, fx)
        else:
            if usetex:
                return (r'$%s%s^{%d}$') % (sign_string,
                                           base,
                                           nearest_long(fx))
            else:
                return (r'$\mathdefault{%s%s^{%d}}$') % (sign_string,
                                                         base,
                                                         nearest_long(fx))


class EngFormatter(Formatter):
    """
    Formats axis values using engineering prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # the unicode for -6 is the greek letter mu
    # commeted here due to bug in pep8
    # (https://github.com/jcrocholl/pep8/issues/271)

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
         -9: "n",
         -6: "\u03bc",
         -3: "m",
          0: "",
          3: "k",
          6: "M",
          9: "G",
         12: "T",
         15: "P",
         18: "E",
         21: "Z",
         24: "Y"
    }

    def __init__(self, unit="", places=None):
        self.unit = unit
        self.places = places

    def __call__(self, x, pos=None):
        s = "%s%s" % (self.format_eng(x), self.unit)
        return self.fix_minus(s)

    def format_eng(self, num):
        """ Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:

        >>> format_eng(0)       # for self.places = 0
        '0'

        >>> format_eng(1000000) # for self.places = 1
        '1.0 M'

        >>> format_eng("-1e-6") # for self.places = 2
        u'-1.00 \u03bc'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """

        dnum = decimal.Decimal(str(num))

        sign = 1

        if dnum < 0:
            sign = -1
            dnum = -dnum

        if dnum != 0:
            pow10 = decimal.Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = decimal.Decimal(0)

        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))

        prefix = self.ENG_PREFIXES[int(pow10)]

        mant = sign * dnum / (10 ** pow10)

        if self.places is None:
            format_str = "%g %s"
        elif self.places == 0:
            format_str = "%i %s"
        elif self.places > 0:
            format_str = ("%%.%if %%s" % self.places)

        formatted = format_str % (mant, prefix)

        return formatted.strip()


class Locator(TickHelper):
    """
    Determine the tick locations;

    Note, you should not use the same locator between different
    :class:`~matplotlib.axis.Axis` because the locator stores references to
    the Axis data and view limits
    """

    # Some automatic tick locators can generate so many ticks they
    # kill the machine when you try and render them.
    # This parameter is set to cause locators to raise an error if too
    # many ticks are generated.
    MAXTICKS = 1000

    def tick_values(self, vmin, vmax):
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated :attr:`axis` simply call
            the Locator instance::

                >>> print((type(loc)))
                <type 'Locator'>
                >>> print((loc()))
                [1, 2, 3, 4]

        """
        raise NotImplementedError('Derived must override')

    def __call__(self):
        """Return the locations of the ticks"""
        # note: some locators return data limits, other return view limits,
        # hence there is no *one* interface to call self.tick_values.
        raise NotImplementedError('Derived must override')

    def raise_if_exceeds(self, locs):
        """raise a RuntimeError if Locator attempts to create more than
           MAXTICKS locs"""
        if len(locs) >= self.MAXTICKS:
            msg = ('Locator attempting to generate %d ticks from %s to %s: ' +
                   'exceeds Locator.MAXTICKS') % (len(locs), locs[0], locs[-1])
            raise RuntimeError(msg)

        return locs

    def view_limits(self, vmin, vmax):
        """
        select a scale for the range from vmin to vmax

        Normally this method is overridden by subclasses to
        change locator behaviour.
        """
        return mtransforms.nonsingular(vmin, vmax)

    def autoscale(self):
        """autoscale the view limits"""
        return self.view_limits(*self.axis.get_view_interval())

    def pan(self, numsteps):
        """Pan numticks (can be positive or negative)"""
        ticks = self()
        numticks = len(ticks)

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        if numticks > 2:
            step = numsteps * abs(ticks[0] - ticks[1])
        else:
            d = abs(vmax - vmin)
            step = numsteps * d / 6.

        vmin += step
        vmax += step
        self.axis.set_view_interval(vmin, vmax, ignore=True)

    def zoom(self, direction):
        "Zoom in/out on axis; if direction is >0 zoom in, else zoom out"

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        interval = abs(vmax - vmin)
        step = 0.1 * interval * direction
        self.axis.set_view_interval(vmin + step, vmax - step, ignore=True)

    def refresh(self):
        """refresh internal information based on current lim"""
        pass


class IndexLocator(Locator):
    """
    Place a tick on every multiple of some base number of points
    plotted, e.g., on every 5th point.  It is assumed that you are doing
    index plotting; i.e., the axis is 0, len(data).  This is mainly
    useful for x ticks.
    """
    def __init__(self, base, offset):
        'place ticks on the i-th data points where (i-offset)%base==0'
        self._base = base
        self.offset = offset

    def __call__(self):
        """Return the locations of the ticks"""
        dmin, dmax = self.axis.get_data_interval()
        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        return self.raise_if_exceeds(
            np.arange(vmin + self.offset, vmax + 1, self._base))


class FixedLocator(Locator):
    """
    Tick locations are fixed.  If nbins is not None,
    the array of possible positions will be subsampled to
    keep the number of ticks <= nbins +1.
    The subsampling will be done so as to include the smallest
    absolute value; for example, if zero is included in the
    array of possibilities, then it is guaranteed to be one of
    the chosen ticks.
    """

    def __init__(self, locs, nbins=None):
        self.locs = np.asarray(locs)
        self.nbins = nbins
        if self.nbins is not None:
            self.nbins = max(self.nbins, 2)

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """"
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        if self.nbins is None:
            return self.locs
        step = max(int(0.99 + len(self.locs) / float(self.nbins)), 1)
        ticks = self.locs[::step]
        for i in range(1, step):
            ticks1 = self.locs[i::step]
            if np.absolute(ticks1).min() < np.absolute(ticks).min():
                ticks = ticks1
        return self.raise_if_exceeds(ticks)


class NullLocator(Locator):
    """
    No ticks
    """

    def __call__(self):
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """"
        Return the locations of the ticks.

        .. note::

            Because the values are Null, vmin and vmax are not used in this
            method.
        """
        return []


class LinearLocator(Locator):
    """
    Determine the tick locations

    The first time this function is called it will try to set the
    number of ticks to make a nice tick partitioning.  Thereafter the
    number of ticks will be fixed so that interactive navigation will
    be nice

    """
    def __init__(self, numticks=None, presets=None):
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
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if (vmin, vmax) in self.presets:
            return self.presets[(vmin, vmax)]

        if self.numticks is None:
            self._set_numticks()

        if self.numticks == 0:
            return []
        ticklocs = np.linspace(vmin, vmax, self.numticks)

        return self.raise_if_exceeds(ticklocs)

    def _set_numticks(self):
        self.numticks = 11  # todo; be smart here; this is just for dev

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if vmin == vmax:
            vmin -= 1
            vmax += 1

        exponent, remainder = divmod(math.log10(vmax - vmin), 1)

        if remainder < 0.5:
            exponent -= 1
        scale = 10 ** (-exponent)
        vmin = math.floor(scale * vmin) / scale
        vmax = math.ceil(scale * vmax) / scale

        return mtransforms.nonsingular(vmin, vmax)


def closeto(x, y):
    if abs(x - y) < 1e-10:
        return True
    else:
        return False


class Base:
    'this solution has some hacks to deal with floating point inaccuracies'
    def __init__(self, base):
        assert(base > 0)
        self._base = base

    def lt(self, x):
        'return the largest multiple of base < x'
        d, m = divmod(x, self._base)
        if closeto(m, 0) and not closeto(m / self._base, 1):
            return (d - 1) * self._base
        return d * self._base

    def le(self, x):
        'return the largest multiple of base <= x'
        d, m = divmod(x, self._base)
        if closeto(m / self._base, 1):  # was closeto(m, self._base)
            #looks like floating point error
            return (d + 1) * self._base
        return d * self._base

    def gt(self, x):
        'return the smallest multiple of base > x'
        d, m = divmod(x, self._base)
        if closeto(m / self._base, 1):
            #looks like floating point error
            return (d + 2) * self._base
        return (d + 1) * self._base

    def ge(self, x):
        'return the smallest multiple of base >= x'
        d, m = divmod(x, self._base)
        if closeto(m, 0) and not closeto(m / self._base, 1):
            return d * self._base
        return (d + 1) * self._base

    def get_base(self):
        return self._base


class MultipleLocator(Locator):
    """
    Set a tick on every integer that is multiple of base in the
    view interval
    """

    def __init__(self, base=1.0):
        self._base = Base(base)

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        vmin = self._base.ge(vmin)
        base = self._base.get_base()
        n = (vmax - vmin + 0.001 * base) // base
        locs = vmin - base + np.arange(n + 3) * base
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest multiples of base that
        contain the data
        """
        vmin = self._base.le(dmin)
        vmax = self._base.ge(dmax)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        return mtransforms.nonsingular(vmin, vmax)


def scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)
    if dv == 0:     # maxabsv == 0 is a special case of this.
        return 1.0, 0.0
        # Note: this should never occur because
        # vmin, vmax should have been checked by nonsingular(),
        # and spread apart if necessary.
    meanv = 0.5 * (vmax + vmin)
    if abs(meanv) / dv < threshold:
        offset = 0
    elif meanv > 0:
        ex = divmod(math.log10(meanv), 1)[0]
        offset = 10 ** ex
    else:
        ex = divmod(math.log10(-meanv), 1)[0]
        offset = -10 ** ex
    ex = divmod(math.log10(dv / n), 1)[0]
    scale = 10 ** ex
    return scale, offset


class MaxNLocator(Locator):
    """
    Select no more than N intervals at nice locations.
    """
    default_params = dict(nbins=10,
                          steps=None,
                          trim=True,
                          integer=False,
                          symmetric=False,
                          prune=None)

    def __init__(self, *args, **kwargs):
        """
        Keyword args:

        *nbins*
            Maximum number of intervals; one less than max number of ticks.

        *steps*
            Sequence of nice numbers starting with 1 and ending with 10;
            e.g., [1, 2, 4, 5, 10]

        *integer*
            If True, ticks will take only integer values.

        *symmetric*
            If True, autoscaling will result in a range symmetric
            about zero.

        *prune*
            ['lower' | 'upper' | 'both' | None]
            Remove edge ticks -- useful for stacked or ganged plots
            where the upper tick of one axes overlaps with the lower
            tick of the axes above it.
            If prune=='lower', the smallest tick will
            be removed.  If prune=='upper', the largest tick will be
            removed.  If prune=='both', the largest and smallest ticks
            will be removed.  If prune==None, no ticks will be removed.

        """
        # I left "trim" out; it defaults to True, and it is not
        # clear that there is any use case for False, so we may
        # want to remove that kwarg.  EF 2010/04/18
        if args:
            kwargs['nbins'] = args[0]
            if len(args) > 1:
                raise ValueError(
                    "Keywords are required for all arguments except 'nbins'")
        self.set_params(**self.default_params)
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        if 'nbins' in kwargs:
            self._nbins = int(kwargs['nbins'])
        if 'trim' in kwargs:
            self._trim = kwargs['trim']
        if 'integer' in kwargs:
            self._integer = kwargs['integer']
        if 'symmetric' in kwargs:
            self._symmetric = kwargs['symmetric']
        if 'prune' in kwargs:
            prune = kwargs['prune']
            if prune is not None and prune not in ['upper', 'lower', 'both']:
                raise ValueError(
                    "prune must be 'upper', 'lower', 'both', or None")
            self._prune = prune
        if 'steps' in kwargs:
            steps = kwargs['steps']
            if steps is None:
                self._steps = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
            else:
                if int(steps[-1]) != 10:
                    steps = list(steps)
                    steps.append(10)
                self._steps = steps
        if 'integer' in kwargs:
            self._integer = kwargs['integer']
        if self._integer:
            self._steps = [n for n in self._steps if divmod(n, 1)[1] < 0.001]

    def bin_boundaries(self, vmin, vmax):
        nbins = self._nbins
        scale, offset = scale_range(vmin, vmax, nbins)
        if self._integer:
            scale = max(1, scale)
        vmin = vmin - offset
        vmax = vmax - offset
        raw_step = (vmax - vmin) / nbins
        scaled_raw_step = raw_step / scale
        best_vmax = vmax
        best_vmin = vmin

        for step in self._steps:
            if step < scaled_raw_step:
                continue
            step *= scale
            best_vmin = step * divmod(vmin, step)[0]
            best_vmax = best_vmin + step * nbins
            if (best_vmax >= vmax):
                break
        if self._trim:
            extra_bins = int(divmod((best_vmax - vmax), step)[0])
            nbins -= extra_bins
        return (np.arange(nbins + 1) * step + best_vmin + offset)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=1e-13,
                                                         tiny=1e-14)
        locs = self.bin_boundaries(vmin, vmax)
        prune = self._prune
        if prune == 'lower':
            locs = locs[1:]
        elif prune == 'upper':
            locs = locs[:-1]
        elif prune == 'both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        if self._symmetric:
            maxabs = max(abs(dmin), abs(dmax))
            dmin = -maxabs
            dmax = maxabs
        dmin, dmax = mtransforms.nonsingular(dmin, dmax, expander=1e-12,
                                                        tiny=1.e-13)
        return np.take(self.bin_boundaries(dmin, dmax), [0, -1])


def decade_down(x, base=10):
    'floor x to the nearest lower decade'
    if x == 0.0:
        return -base
    lx = np.floor(np.log(x) / np.log(base))
    return base ** lx


def decade_up(x, base=10):
    'ceil x to the nearest higher decade'
    if x == 0.0:
        return base
    lx = np.ceil(np.log(x) / np.log(base))
    return base ** lx


def nearest_long(x):
    if x == 0:
        return long(0)
    elif x > 0:
        return long(x + 0.5)
    else:
        return long(x - 0.5)


def is_decade(x, base=10):
    if not np.isfinite(x):
        return False
    if x == 0.0:
        return True
    lx = np.log(np.abs(x)) / np.log(base)
    return is_close_to_int(lx)


def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - nearest_long(x)) < 1e-10


class LogLocator(Locator):
    """
    Determine the tick locations for log axes
    """

    def __init__(self, base=10.0, subs=[1.0], numdecs=4, numticks=15):
        """
        place ticks on the location= base**i*subs[j]
        """
        self.base(base)
        self.subs(subs)
        self.numticks = numticks
        self.numdecs = numdecs

    def base(self, base):
        """
        set the base of the log scaling (major tick every base**i, i integer)
        """
        self._base = base + 0.0

    def subs(self, subs):
        """
        set the minor ticks the log scaling every base**i*subs[j]
        """
        if subs is None:
            self._subs = None  # autosub
        else:
            self._subs = np.asarray(subs) + 0.0

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        b = self._base
        # dummy axis has no axes attribute
        if hasattr(self.axis, 'axes') and self.axis.axes.name == 'polar':
            vmax = math.ceil(math.log(vmax) / math.log(b))
            decades = np.arange(vmax - self.numdecs, vmax)
            ticklocs = b ** decades

            return ticklocs

        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError(
                    "Data has no positive values, and therefore can not be "
                    "log-scaled.")

        vmin = math.log(vmin) / math.log(b)
        vmax = math.log(vmax) / math.log(b)

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        numdec = math.floor(vmax) - math.ceil(vmin)

        if self._subs is None:  # autosub
            if numdec > 10:
                subs = np.array([1.0])
            elif numdec > 6:
                subs = np.arange(2.0, b, 2.0)
            else:
                subs = np.arange(2.0, b)
        else:
            subs = self._subs

        stride = 1
        while numdec / stride + 1 > self.numticks:
            stride += 1

        decades = np.arange(math.floor(vmin) - stride,
                            math.ceil(vmax) + 2 * stride, stride)
        if hasattr(self, '_transform'):
            ticklocs = self._transform.inverted().transform(decades)
            if len(subs) > 1 or (len(subs == 1) and subs[0] != 1.0):
                ticklocs = np.ravel(np.outer(subs, ticklocs))
        else:
            if len(subs) > 1 or (len(subs == 1) and subs[0] != 1.0):
                ticklocs = []
                for decadeStart in b ** decades:
                    ticklocs.extend(subs * decadeStart)
            else:
                ticklocs = b ** decades

        return self.raise_if_exceeds(np.asarray(ticklocs))

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'
        b = self._base

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if self.axis.axes.name == 'polar':
            vmax = math.ceil(math.log(vmax) / math.log(b))
            vmin = b ** (vmax - self.numdecs)
            return vmin, vmax

        minpos = self.axis.get_minpos()

        if minpos <= 0 or not np.isfinite(minpos):
            raise ValueError(
                "Data has no positive values, and therefore can not be "
                "log-scaled.")

        if vmin <= minpos:
            vmin = minpos

        if not is_decade(vmin, self._base):
            vmin = decade_down(vmin, self._base)
        if not is_decade(vmax, self._base):
            vmax = decade_up(vmax, self._base)

        if vmin == vmax:
            vmin = decade_down(vmin, self._base)
            vmax = decade_up(vmax, self._base)
        result = mtransforms.nonsingular(vmin, vmax)
        return result


class SymmetricalLogLocator(Locator):
    """
    Determine the tick locations for log axes
    """

    def __init__(self, transform, subs=None):
        """
        place ticks on the location= base**i*subs[j]
        """
        self._transform = transform
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def __call__(self):
        'Return the locations of the ticks'
        # Note, these are untransformed coordinates
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        b = self._transform.base
        t = self._transform.linthresh

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # The domain is divided into three sections, only some of
        # which may actually be present.
        #
        # <======== -t ==0== t ========>
        # aaaaaaaaa    bbbbb   ccccccccc
        #
        # a) and c) will have ticks at integral log positions.  The
        # number of ticks needs to be reduced if there are more
        # than self.numticks of them.
        #
        # b) has a tick at 0 and only 0 (we assume t is a small
        # number, and the linear segment is just an implementation
        # detail and not interesting.)
        #
        # We could also add ticks at t, but that seems to usually be
        # uninteresting.
        #
        # "simple" mode is when the range falls entirely within (-t,
        # t) -- it should just display (vmin, 0, vmax)

        has_a = has_b = has_c = False
        if vmin < -t:
            has_a = True
            if vmax > -t:
                has_b = True
                if vmax > t:
                    has_c = True
        elif vmin < 0:
            if vmax > 0:
                has_b = True
                if vmax > t:
                    has_c = True
            else:
                return [vmin, vmax]
        elif vmin < t:
            if vmax > t:
                has_b = True
                has_c = True
            else:
                return [vmin, vmax]
        else:
            has_c = True

        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(b))
            hi = np.ceil(np.log(hi) / np.log(b))
            return lo, hi

        # First, calculate all the ranges, so we can determine striding
        if has_a:
            if has_b:
                a_range = get_log_range(t, -vmin + 1)
            else:
                a_range = get_log_range(-vmax, -vmin + 1)
        else:
            a_range = (0, 0)

        if has_c:
            if has_b:
                c_range = get_log_range(t, vmax + 1)
            else:
                c_range = get_log_range(vmin, vmax + 1)
        else:
            c_range = (0, 0)

        total_ticks = (a_range[1] - a_range[0]) + (c_range[1] - c_range[0])
        if has_b:
            total_ticks += 1
        stride = max(np.floor(float(total_ticks) / (self.numticks - 1)), 1)

        decades = []
        if has_a:
            decades.extend(-1 * (b ** (np.arange(a_range[0], a_range[1],
                                                 stride)[::-1])))

        if has_b:
            decades.append(0.0)

        if has_c:
            decades.extend(b ** (np.arange(c_range[0], c_range[1], stride)))

        # Add the subticks if requested
        if self._subs is None:
            subs = np.arange(2.0, b)
        else:
            subs = np.asarray(self._subs)

        if len(subs) > 1 or subs[0] != 1.0:
            ticklocs = []
            for decade in decades:
                ticklocs.extend(subs * decade)
        else:
            ticklocs = decades

        return self.raise_if_exceeds(np.array(ticklocs))

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'
        b = self._transform.base
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if not is_decade(abs(vmin), b):
            if vmin < 0:
                vmin = -decade_up(-vmin, b)
            else:
                vmin = decade_down(vmin, b)
        if not is_decade(abs(vmax), b):
            if vmax < 0:
                vmax = -decade_down(-vmax, b)
            else:
                vmax = decade_up(vmax, b)

        if vmin == vmax:
            if vmin < 0:
                vmin = -decade_up(-vmin, b)
                vmax = -decade_down(-vmax, b)
            else:
                vmin = decade_down(vmin, b)
                vmax = decade_up(vmax, b)
        result = mtransforms.nonsingular(vmin, vmax)
        return result


class AutoLocator(MaxNLocator):
    def __init__(self):
        MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])


class AutoMinorLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks. Assumes the scale is linear and major ticks are
    evenly spaced.
    """
    def __init__(self, n=None):
        """
        *n* is the number of subdivisions of the interval between
        major ticks; e.g., n=2 will place a single minor tick midway
        between major ticks.

        If *n* is omitted or None, it will be set to 5 or 4.
        """
        self.ndivs = n

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        try:
            majorstep = majorlocs[1] - majorlocs[0]
        except IndexError:
            # Need at least two major ticks to find minor tick locations
            # TODO: Figure out a way to still be able to display minor
            # ticks without two major ticks visible. For now, just display
            # no ticks at all.
            majorstep = 0

        if self.ndivs is None:
            if majorstep == 0:
                # TODO: Need a better way to figure out ndivs
                ndivs = 1
            else:
                x = int(round(10 ** (np.log10(majorstep) % 1)))
                if x in [1, 5, 10]:
                    ndivs = 5
                else:
                    ndivs = 4
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs

        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if len(majorlocs) > 0:
            t0 = majorlocs[0]
            tmin = ((vmin - t0) // minorstep + 1) * minorstep
            tmax = ((vmax - t0) // minorstep + 1) * minorstep
            locs = np.arange(tmin, tmax, minorstep) + t0
            cond = np.abs((locs - t0) % majorstep) > minorstep / 10.0
            locs = locs.compress(cond)
        else:
            locs = []

        return self.raise_if_exceeds(np.array(locs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


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
        return self.raise_if_exceeds(self._locator())

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

    def refresh(self):
        'refresh internal information based on current lim'
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        d = abs(vmax - vmin)
        self._locator = self.get_locator(d)

    def view_limits(self, vmin, vmax):
        'Try to choose the view limits intelligently'

        d = abs(vmax - vmin)
        self._locator = self.get_locator(d)
        return self._locator.view_limits(vmin, vmax)

    def get_locator(self, d):
        'pick the best locator based on a distance'
        d = abs(d)
        if d <= 0:
            locator = MultipleLocator(0.2)
        else:

            try:
                ld = math.log10(d)
            except OverflowError:
                raise RuntimeError('AutoLocator illegal data interval range')

            fld = math.floor(ld)
            base = 10 ** fld

            #if ld==fld:  base = 10**(fld-1)
            #else:        base = 10**fld

            if d >= 5 * base:
                ticksize = base
            elif d >= 2 * base:
                ticksize = base / 2.0
            else:
                ticksize = base / 5.0
            locator = MultipleLocator(ticksize)

        return locator


__all__ = ('TickHelper', 'Formatter', 'FixedFormatter',
           'NullFormatter', 'FuncFormatter', 'FormatStrFormatter',
           'ScalarFormatter', 'LogFormatter', 'LogFormatterExponent',
           'LogFormatterMathtext', 'Locator', 'IndexLocator',
           'FixedLocator', 'NullLocator', 'LinearLocator',
           'LogLocator', 'AutoLocator', 'MultipleLocator',
           'MaxNLocator', 'AutoMinorLocator',)
