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


   ax.get_xaxis().get_major_formatter().use_offset = False

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

:class:`SymmetricalLogLocator`
    locator for use with with the symlog norm, works like the `LogLocator` for
    the part outside of the threshold and add 0 if inside the limits

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

:class:`StrMethodFormatter`
    Use string `format` method

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

from matplotlib.externals import six

import decimal
import locale
import math
import warnings

import numpy as np
from matplotlib import rcParams
from matplotlib import cbook
from matplotlib import transforms as mtransforms

if six.PY3:
    long = int


def _mathdefault(s):
    """
    For backward compatibility, in classic mode we display
    sub/superscripted text in a mathdefault block.  As of 2.0, the
    math font already matches the default font, so we don't need to do
    that anymore.
    """
    assert s[0] == s[-1] == "$"
    if rcParams["_internal.classic_mode"] and not rcParams["text.usetex"]:
        return r"$\mathdefault{{{}}}$".format(s[1:-1])
    else:
        return s


def _unicode_minus(s):
    return s.replace("-", "\N{MINUS SIGN}")


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

    def get_tick_space(self):
        # Just use the long-standing default of nbins==9
        return 9


class _TickHelper(object):
    def __init__(self):
        self.axis = None

    def set_axis(self, axis_or_kw):
        if isinstance(axis_or_kw, dict):
            self.axis = _DummyAxis(**axis_or_kw)
        else:
            self.axis = axis_or_kw


class Formatter(_TickHelper):
    def __init__(self):
        super(Formatter, self).__init__()
        self._locs = []
        self._offset_text = ""

    @property
    def locs(self):
        return self._locs

    @locs.setter
    def locs(self, locs):
        self._locs = np.asarray(locs)

    @property
    def offset_text(self):
        return self._offset_text

    @offset_text.setter
    def offset_text(self, s):
        self._offset_text = s

    def format_for_tick(self, value, pos=None):
        return ""

    def format_for_cursor(self, value):
        return self.format_for_tick(value)


class NullFormatter(Formatter):
    """Always return the empty string.
    """


class FixedFormatter(Formatter):
    """Return fixed strings for tick labels.
    """

    def __init__(self, seq):
        super(FixedFormatter, self).__init__()
        self._seq = seq

    def format_for_tick(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*.
        """
        return self._seq[pos] if pos < len(self._seq) else ""


class FuncFormatter(Formatter):
    """User defined function for formatting.

    The function should take in two inputs (tick value *x* and position *pos*)
    and return a string.
    """

    def __init__(self, func):
        super(FuncFormatter, self).__init__()
        self.format_for_tick = func


class FormatStrFormatter(Formatter):
    """Use an old-style ('%' operator) format string to format the tick.
    """

    def __init__(self, fmt):
        super(FormatStrFormatter, self).__init__()
        self._fmt = fmt

    def format_for_tick(self, x, pos=None):
        return self._fmt % x


class StrMethodFormatter(Formatter):
    """Use a new-style format string (as used by `str.format()`) to
    format the tick.  The field formatting must be labeled `x` and/or
    `pos`.
    """

    def __init__(self, fmt):
        super(StrMethodFormatter, self).__init__()
        self._fmt = fmt

    def format_for_tick(self, x, pos=None):
        return self._fmt.format(x=x, pos=pos)


class ScalarFormatter(Formatter):
    """Tick location is a plain old number.  If use_offset==True
    and the data range is much smaller than the data average, then
    an offset will be determined such that the tick labels are
    meaningful. Scientific notation is used for data < 10^-n or
    data >= 10^m, where n and m are the power limits set using
    set_powerlimits((n,m)). The defaults for these are controlled by the
    axes.formatter.limits rc parameter.
    """

    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        super(ScalarFormatter, self).__init__()
        self._use_offset = (useOffset if useOffset is not None
                            else rcParams["axes.formatter.useoffset"])
        self._usetex = rcParams["text.usetex"]
        self._use_mathtext = (useMathText if useMathText is not None
                              else rcParams["axes.formatter.use_mathtext"])
        self._use_locale = (useLocale if useLocale is not None
                            else rcParams["axes.formatter.use_locale"])
        self._scientific = True
        self._powerlimits = rcParams['axes.formatter.limits']
        self._offset = 0
        self._oom = self._tick_precision = self._cursor_precision = None

    def _format_maybe_locale(self, fmt, *args):
        return (locale.format_string(fmt, *args) if self._use_locale
                else fmt % args)

    @property
    def locs(self):
        return self._locs

    @locs.setter
    def locs(self, locs):
        self._locs = np.asarray(locs)
        self._update()

    @property
    def use_offset(self):
        return self._use_offset

    @use_offset.setter
    def use_offset(self, b):
        self._use_offset = b
        self._update()

    @property
    def use_locale(self):
        return self._use_locale

    @use_locale.setter
    def use_locale(self, b):
        self._use_locale = b
        self._update()

    @property
    def scientific(self):
        return self._scientific

    @scientific.setter
    def scientific(self, b):
        self._scientific = b
        self._update()

    @property
    def powerlimits(self):
        return self._powerlimits

    @powerlimits.setter
    def powerlimits(self, lims):
        self._powerlimits = lims
        self._update()

    def _update(self):
        self._set_offset()
        self._set_oom()
        self._set_precision()

    def _set_offset(self):
        locs = self.locs
        if not self._use_offset:
            self._offset = 0
            return
        # Restrict to visible ticks.
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        if not len(locs):
            self._offset = 0
            return
        lmin, lmax = locs.min(), locs.max()
        # min, max comparing absolute values (we want division to round towards
        # zero so we work on absolute values).
        abs_min, abs_max = sorted(map(abs, [lmin, lmax]))
        # Only use offset if there are at least two ticks, every tick has the
        # same sign, and if the span is small compared to the absolute values.
        if (lmin == lmax or lmin <= 0 <= lmax or
                (abs_max - abs_min) / abs_max >= 1e-2):
            self._offset = 0
            return
        sign = math.copysign(1, lmin)
        # What is the smallest power of ten such that abs_min and abs_max are
        # equal up to that precision?
        oom = 10. ** math.ceil(math.log10(abs_max))
        while True:
            if abs_min // oom != abs_max // oom:
                oom *= 10
                break
            oom /= 10
        if (abs_max - abs_min) / oom <= 1e-2:
            # Handle the case of straddling a multiple of a large power of ten
            # (relative to the span).
            # What is the smallest power of ten such that abs_min and abs_max
            # at most 1 apart?
            oom = 10. ** math.ceil(math.log10(abs_max))
            while True:
                if abs_max // oom - abs_min // oom > 1:
                    oom *= 10
                    break
                oom /= 10
        self._offset = sign * (abs_max // oom) * oom

    def _set_oom(self):
        # If scientific notation is to be used, find the appropriate exponent.
        # If using an offset, find the exponent after applying the offset.
        if self._scientific:
            locs = self.locs
            vmin, vmax = sorted(self.axis.get_view_interval())
            # Restrict to visible ticks.
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            if len(locs):
                deltas = np.abs(locs - self._offset)
                # Don't take tiny, nonzero tick values into account.
                deltas = deltas[deltas > 1e-10 * deltas.max()]
                if len(deltas):
                    oom = int(math.floor(math.log10(deltas.min())))
                    self._oom = (oom if (oom <= self._powerlimits[0] or
                                         oom >= self._powerlimits[1])
                                else 0)
                else:
                    self._oom = 0
            else:
                self._oom = 0
        else:
            self._oom = 0

    def _set_precision(self):
        locs = self.locs
        vmin, vmax = sorted(self.axis.get_view_interval())
        # Restrict to visible ticks.
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        # Tick precision.
        if len(locs):
            ticks = np.abs(locs - self._offset) / 10. ** self._oom
            # Don't take tiny, nonzero tick values into account.
            ticks = ticks[ticks > 1e-10 * ticks.max()]
            if len(ticks):
                thresh = 10. ** (math.floor(math.log10(ticks.min())) - 3)
                precision = 0
                while (np.abs(np.round(ticks, precision) - ticks).max() >
                       thresh):
                    precision += 1
                # Work around rounding errors, e.g. test_formatter_large_small:
                # increase scale and recompute precision.
                if (self._oom and
                        np.abs(np.round(ticks, precision)).min() >= 10):
                    self._oom += 1
                    self._set_precision()
                    return
                self._tick_precision = precision
            else:
                self._tick_precision = 0
        else:
            self._tick_precision = 0
        # Cursor precision.
        self._cursor_precision = (
            3 - int(math.floor(math.log10(vmax - vmin)))) if vmax > vmin else 0

    @property
    def offset_text(self):
        if self._oom or self._offset:
            if self._offset:
                offset_str = self._format_offset()
            else:
                offset_str = ""
            if self._oom:
                if self._use_mathtext or rcParams["text.usetex"]:
                    sci_not_str = "$10^{{{}}}$".format(self._oom)
                else:
                    sci_not_str = "1e{}".format(self._oom)
            else:
                sci_not_str = ""
            if self._use_mathtext or rcParams["text.usetex"]:
                assert offset_str[0] == offset_str[-1] == "$"
                if sci_not_str:
                    assert sci_not_str[0] == sci_not_str[-1] == "$"
                    s = r"$\times {} {}$".format(
                        sci_not_str[1:-1], offset_str[1:-1])
                else:
                    s = offset_str
                return _mathdefault(s)
            else:
                if sci_not_str:
                    s = "\N{MULTIPLICATION SIGN}{}{}".format(
                        sci_not_str, offset_str)
                else:
                    s = offset_str
                return _unicode_minus(s)
        else:
            return ""

    def _format_offset(self):
        offset = self._offset
        # How many significant digits are needed to represent offset to
        # 10 ** (oom - tick_precision)?
        precision = (
            (int(math.floor(math.log10(abs(offset)))) if offset else 0) +
            self._tick_precision - self._oom + 1)
        if .0001 <= abs(offset) < 10000 or offset == 0:
            # %g doesn't use scientific notation in this range.
            s = (self._format_maybe_locale("%#-+.*g", precision, offset).
                 rstrip(locale.localeconv()["decimal_point"]
                        if self._use_locale else "."))
            if self._use_mathtext or rcParams["text.usetex"]:
                return "${}$".format(s)
            else:
                return s
        exp = int(math.floor(math.log10(abs(offset))))
        significand = offset / 10. ** exp
        # 1 digit before decimal point, (precision - 1) after.
        significand_s = self._format_maybe_locale(
            "%-+.*f", precision - 1, significand)
        if self._use_mathtext or rcParams["text.usetex"]:
            return r"${} \times 10^{{{}}}$".format(significand_s, exp)
        else:
            return "{}e{}".format(significand_s, exp)

    def format_for_tick(self, value, pos=None):
        # Get rid of signed zeros (sic).
        scaled = round((value - self._offset) / 10. ** self._oom,
                       self._tick_precision)
        if scaled == 0:
            scaled = 0
        s = self._format_maybe_locale("%-.*f", self._tick_precision, scaled)
        if self._use_mathtext or rcParams["text.usetex"]:
            return _mathdefault("${}$".format(s))
        else:
            return _unicode_minus(s)

    def format_for_cursor(self, value):
        # How many significant digits needed to represent value to
        # 10 ** -cursor_precision?
        precision = (
            math.floor(math.log10(abs(value))) + self._cursor_precision
            if value else self._cursor_precision) + 1
        return _unicode_minus(
            self._format_maybe_locale("%#-.*g", precision, value))


class LogFormatter(Formatter):
    """Format values for log axis.
    """

    def __init__(self, base=10., label_minor=False):
        """
        *base* is used to locate the decade tick, which will be the only
        one to be labeled if *label_minor* is ``False``.
        """
        super(LogFormatter, self).__init__()
        self.base = base
        self.label_minor = label_minor

    def format_for_tick(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        vmin, vmax = sorted(self.axis.get_view_interval())
        d = vmax - vmin
        b = self.base
        if x == 0.0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        is_decade = is_close_to_int(fx)
        if not (is_decade or self.label_minor):
            s = ""
        elif x > 10000:
            s = '%1.0e' % x
        elif x < 1:
            s = '%1.0e' % x
        else:
            s = self._pprint_val(x, d)
        if sign == -1:
            s = '-%s' % s
        return _unicode_minus(s)

    def _pprint_val(self, x, d):
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

        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            if exponent:
                s = '%se%d' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s

    def format_for_cursor(self, value):
        'return a short formatted string representation of a number'
        return '%-12g' % value


class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis; using ``exponent = log_base(value)``
    """

    def format_for_tick(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        d = abs(vmax - vmin)
        b = self.base
        if x == 0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        is_decade = is_close_to_int(fx)
        if not (is_decade or self.label_minor):
            s = ""
        elif abs(fx) < 1 or abs(fx) > 10000:
            s = '%1.0g' % fx
        else:
            fd = math.log(abs(d)) / math.log(b)
            s = self._pprint_val(fx, fd)
        if sign == -1:
            s = '-%s' % s
        return _unicode_minus(s)


class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis; using ``exponent = log_base(value)``
    """

    def format_for_tick(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        b = self.base

        # only label the decades
        if x == 0:
            return _mathdefault("$0$")

        fx = math.log(abs(x)) / math.log(b)
        is_decade = is_close_to_int(fx)

        sign_string = '-' if x < 0 else ""

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        if not (is_decade or self.label_minor):
            return ""
        elif not is_decade:
            return _mathdefault(r"$%s%s^{%.2f}$" %
                                (sign_string, base, fx))
        else:
            return _mathdefault(r"$%s%s^{%d}$" %
                                (sign_string, base, round(fx)))


class LogitFormatter(Formatter):
    ""'Probability formatter (using Math text)""'
    def format_for_tick(self, x, pos=None):
        s = ""
        if 0.01 <= x <= 0.99:
            s = '{:.2f}'.format(x)
        elif x < 0.01:
            if is_decade(x):
                s = _mathdefault('$10^{{{:.0f}}}$'.format(np.log10(x)))
            else:
                s = _mathdefault('${:.5f}$'.format(x))
        else:  # x > 0.99
            if is_decade(1-x):
                s = _mathdefault('$1-10^{{{:.0f}}}$'.format(np.log10(1-x)))
            else:
                s = _mathdefault('$1-{:.5f}$'.format(1-x))
        return s

    def format_for_cursor(self, value):
        'return a short formatted string representation of a number'
        return '%-12g' % value


class EngFormatter(Formatter):
    """
    Formats axis values using engineering prefixes to represent powers of 1000,
    plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
         -9: "n",
         -6: "\N{MICRO SIGN}",
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
        super(EngFormatter, self).__init__()
        self.unit = unit
        self.places = places

    def format_for_tick(self, x, pos=None):
        s = "%s%s" % (self._format_eng(x), self.unit)
        if rcParams["text.usetex"]:
            return _mathdefault(s)
        else:
            return _unicode_minus(s)

    def _format_eng(self, num):
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


class Locator(_TickHelper):
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

    def set_params(self, **kwargs):
        """
        Do nothing, and rase a warning. Any locator class not supporting the
        set_params() function will call this.
        """
        warnings.warn("'set_params()' not defined for locator of type " +
                      str(type(self)))

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
        super(IndexLocator, self).__init__()
        self._base = base
        self.offset = offset

    def set_params(self, base=None, offset=None):
        """Set parameters within this locator"""
        if base is not None:
            self._base = base
        if offset is not None:
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
        super(FixedLocator, self).__init__()
        self.locs = np.asarray(locs)
        self.nbins = nbins
        if self.nbins is not None:
            self.nbins = max(self.nbins, 2)

    def set_params(self, nbins=None):
        """Set parameters within this locator."""
        if nbins is not None:
            self.nbins = nbins

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
        super(LinearLocator, self).__init__()
        self.numticks = numticks
        if presets is None:
            self.presets = {}
        else:
            self.presets = presets

    def set_params(self, numticks=None, presets=None):
        """Set parameters within this locator."""
        if presets is not None:
            self.presets = presets
        if numticks is not None:
            self.numticks = numticks

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

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
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


class Base(object):
    'this solution has some hacks to deal with floating point inaccuracies'
    def __init__(self, base):
        if base <= 0:
            raise ValueError("'base' must be positive")
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
        super(MultipleLocator, self).__init__()
        self._base = Base(base)

    def set_params(self, base):
        """Set parameters within this locator."""
        if base is not None:
            self._base = base

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
        if rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = self._base.le(dmin)
            vmax = self._base.ge(dmax)
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        else:
            vmin = dmin
            vmax = dmax

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
            Maximum number of intervals; one less than max number of
            ticks.  If the string `'auto'`, the number of bins will be
            automatically determined based on the length of the axis.

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
        super(MaxNLocator, self).__init__()
        if args:
            kwargs['nbins'] = args[0]
            if len(args) > 1:
                raise ValueError(
                    "Keywords are required for all arguments except 'nbins'")
        self.set_params(**self.default_params)
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """Set parameters within this locator."""
        if 'nbins' in kwargs:
            self._nbins = kwargs['nbins']
            if self._nbins != 'auto':
                self._nbins = int(self._nbins)
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
        if nbins == 'auto':
            nbins = max(min(self.axis.get_tick_space(), 9), 1)
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
        if rcParams['axes.autolimit_mode'] == 'round_numbers':
            if self._symmetric:
                maxabs = max(abs(dmin), abs(dmax))
                dmin = -maxabs
                dmax = maxabs

        dmin, dmax = mtransforms.nonsingular(dmin, dmax, expander=1e-12,
                                                        tiny=1.e-13)

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
            return np.take(self.bin_boundaries(dmin, dmax), [0, -1])
        else:
            return dmin, dmax


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
    return abs(x - round(x)) < 1e-10


class LogLocator(Locator):
    """
    Determine the tick locations for log axes
    """

    def __init__(self, base=10.0, subs=[1.0], numdecs=4, numticks=15):
        """
        place ticks on the location= base**i*subs[j]
        """
        super(LogLocator, self).__init__()
        self.base(base)
        self.subs(subs)
        self.numticks = numticks
        self.numdecs = numdecs

    def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
        """Set parameters within this locator."""
        if base is not None:
            self.base = base
        if subs is not None:
            self.subs = subs
        if numdecs is not None:
            self.numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks

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

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
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
        super(SymmetricalLogLocator, self).__init__()
        self._transform = transform
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """Set parameters within this locator."""
        if numticks is not None:
            self.numticks = numticks
        if subs is not None:
            self._subs = subs

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

        if rcParams['axes.autolimit_mode'] == 'round_numbers':
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


class LogitLocator(Locator):
    """
    Determine the tick locations for logit axes
    """

    def __init__(self, minor=False):
        """
        place ticks on the logit locations
        """
        super(LogitLocator, self).__init__()
        self.minor = minor

    def set_params(self, minor=None):
        """Set parameters within this locator."""
        if minor is not None:
            self.minor = minor

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        # dummy axis has no axes attribute
        if hasattr(self.axis, 'axes') and self.axis.axes.name == 'polar':
            raise NotImplementedError('Polar axis cannot be logit scaled yet')

        # what to do if a window beyond ]0, 1[ is chosen
        if vmin <= 0.0:
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            if (vmin <= 0.0) or (not np.isfinite(vmin)):
                raise ValueError(
                    "Data has no values in ]0, 1[ and therefore can not be "
                    "logit-scaled.")

        # NOTE: for vmax, we should query a property similar to get_minpos, but
        # related to the maximal, less-than-one data point. Unfortunately,
        # get_minpos is defined very deep in the BBox and updated with data,
        # so for now we use the trick below.
        if vmax >= 1.0:
            if self.axis is not None:
                vmax = 1 - self.axis.get_minpos()

            if (vmax >= 1.0) or (not np.isfinite(vmax)):
                raise ValueError(
                    "Data has no values in ]0, 1[ and therefore can not be "
                    "logit-scaled.")

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        vmin = np.log10(vmin / (1 - vmin))
        vmax = np.log10(vmax / (1 - vmax))

        decade_min = np.floor(vmin)
        decade_max = np.ceil(vmax)

        # major ticks
        if not self.minor:
            ticklocs = []
            if (decade_min <= -1):
                expo = np.arange(decade_min, min(0, decade_max + 1))
                ticklocs.extend(list(10**expo))
            if (decade_min <= 0) and (decade_max >= 0):
                ticklocs.append(0.5)
            if (decade_max >= 1):
                expo = -np.arange(max(1, decade_min), decade_max + 1)
                ticklocs.extend(list(1 - 10**expo))

        # minor ticks
        else:
            ticklocs = []
            if (decade_min <= -2):
                expo = np.arange(decade_min, min(-1, decade_max))
                newticks = np.outer(np.arange(2, 10), 10**expo).ravel()
                ticklocs.extend(list(newticks))
            if (decade_min <= 0) and (decade_max >= 0):
                ticklocs.extend([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
            if (decade_max >= 2):
                expo = -np.arange(max(2, decade_min), decade_max + 1)
                newticks = 1 - np.outer(np.arange(2, 10), 10**expo).ravel()
                ticklocs.extend(list(newticks))

        return self.raise_if_exceeds(np.array(ticklocs))


class AutoLocator(MaxNLocator):
    def __init__(self):
        if rcParams['_internal.classic_mode']:
            nbins = 9
        else:
            nbins = 'auto'
        super(AutoLocator, self).__init__(nbins=nbins, steps=[1, 2, 5, 10])


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
        super(AutoMinorLocator, self).__init__()
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
                x = int(np.round(10 ** (np.log10(majorstep) % 1)))
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
        super(OldAutoLocator, self).__init__()
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


__all__ = ['Formatter', 'FixedFormatter', 'NullFormatter', 'FuncFormatter',
           'FormatStrFormatter', 'StrMethodFormatter', 'ScalarFormatter',
           'LogFormatter', 'LogFormatterExponent', 'LogFormatterMathtext',
           'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator',
           'LinearLocator', 'LogLocator', 'AutoLocator', 'MultipleLocator',
           'MaxNLocator', 'AutoMinorLocator', 'SymmetricalLogLocator']
