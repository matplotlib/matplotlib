#!/usr/bin/env python
"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime`, the add-on modules :mod:`pytz` and
:mod:`dateutil`.  :class:`datetime` objects are converted to floating point
numbers which represent time in days since 0001-01-01 UTC, plus 1.  For
example, 0001-01-01, 06:00 is 1.25, not 0.25.  The helper functions
:func:`date2num`, :func:`num2date` and :func:`drange` are used to facilitate
easy conversion to and from :mod:`datetime` and numeric ranges.

.. note::

   Like Python's datetime, mpl uses the Gregorian calendar for all
   conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and mpl give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [31]:date(2006,4,1).toordinal() - date(1,1,1).toordinal()
     Out[31]:732401


A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

All the matplotlib date converters, tickers and formatters are
timezone aware, and the default timezone is given by the timezone
parameter in your :file:`matplotlibrc` file.  If you leave out a
:class:`tz` timezone instance, the default from your rc file will be
assumed.  If you want to use a custom time zone, pass a
:class:`pytz.timezone` instance with the tz keyword argument to
:func:`num2date`, :func:`plot_date`, and any custom date tickers or
locators you create.  See `pytz <http://pytz.sourceforge.net>`_ for
information on :mod:`pytz` and timezone handling.

The `dateutil module <http://labix.org/python-dateutil>`_ provides
additional code to handle date ticking, making it easy to place ticks
on any kinds of dates.  See examples below.

Date tickers
------------

Most of the date tickers can locate single or multiple values.  For
example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on mondays and saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

Here are all the date tickers:

    * :class:`MinuteLocator`: locate minutes

    * :class:`HourLocator`: locate hours

    * :class:`DayLocator`: locate specifed days of the month

    * :class:`WeekdayLocator`: Locate days of the week, e.g., MO, TU

    * :class:`MonthLocator`: locate months, e.g., 7 for july

    * :class:`YearLocator`: locate years that are multiples of base

    * :class:`RRuleLocator`: locate using a
      :class:`matplotlib.dates.rrulewrapper`.  The
      :class:`rrulewrapper` is a simple wrapper around a
      :class:`dateutil.rrule` (`dateutil
      <http://labix.org/python-dateutil>`_) which allow almost
      arbitrary date tick specifications.  See `rrule example
      <../examples/pylab_examples/date_demo_rrule.html>`_.

    * :class:`AutoDateLocator`: On autoscale, this class picks the best
      :class:`MultipleDateLocator` to set the view limits and the tick
      locations.

Date formatters
---------------

Here all all the date formatters:

    * :class:`AutoDateFormatter`: attempts to figure out the best format
      to use.  This is most useful when used with the :class:`AutoDateLocator`.

    * :class:`DateFormatter`: use :func:`strftime` format strings

    * :class:`IndexDateFormatter`: date plots with implicit *x*
      indexing.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange, zip

import re
import time
import math
import datetime

import warnings


from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
                            MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
                            SECONDLY)
from dateutil.relativedelta import relativedelta
import dateutil.parser
import numpy as np


import matplotlib
import matplotlib.units as units
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker


__all__ = ('date2num', 'num2date', 'drange', 'epoch2num',
           'num2epoch', 'mx2num', 'DateFormatter',
           'IndexDateFormatter', 'AutoDateFormatter', 'DateLocator',
           'RRuleLocator', 'AutoDateLocator', 'YearLocator',
           'MonthLocator', 'WeekdayLocator',
           'DayLocator', 'HourLocator', 'MinuteLocator',
           'SecondLocator', 'MicrosecondLocator',
           'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU',
           'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY',
           'HOURLY', 'MINUTELY', 'SECONDLY', 'MICROSECONDLY', 'relativedelta',
           'seconds', 'minutes', 'hours', 'weeks')


# Make a simple UTC instance so we don't always have to import
# pytz.  From the python datetime library docs:

class _UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return datetime.timedelta(0)

UTC = _UTC()


def _get_rc_timezone():
    s = matplotlib.rcParams['timezone']
    if s == 'UTC':
        return UTC
    import pytz
    return pytz.timezone(s)

MICROSECONDLY = SECONDLY + 1
HOURS_PER_DAY = 24.
MINUTES_PER_DAY = 60. * HOURS_PER_DAY
SECONDS_PER_DAY = 60. * MINUTES_PER_DAY
MUSECONDS_PER_DAY = 1e6 * SECONDS_PER_DAY
SEC_PER_MIN = 60
SEC_PER_HOUR = 3600
SEC_PER_DAY = SEC_PER_HOUR * 24
SEC_PER_WEEK = SEC_PER_DAY * 7
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = (
    MO, TU, WE, TH, FR, SA, SU)
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)


def _to_ordinalf(dt):
    """
    Convert :mod:`datetime` to the Gregorian date as UTC float days,
    preserving hours, minutes, seconds and microseconds.  Return value
    is a :func:`float`.
    """

    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta

    base = float(dt.toordinal())
    if hasattr(dt, 'hour'):
        base += (dt.hour / HOURS_PER_DAY + dt.minute / MINUTES_PER_DAY +
                 dt.second / SECONDS_PER_DAY +
                 dt.microsecond / MUSECONDS_PER_DAY
                 )
    return base


# a version of _to_ordinalf that can operate on numpy arrays
_to_ordinalf_np_vectorized = np.vectorize(_to_ordinalf)


def _from_ordinalf(x, tz=None):
    """
    Convert Gregorian float of the date, preserving hours, minutes,
    seconds and microseconds.  Return value is a :class:`datetime`.
    """
    if tz is None:
        tz = _get_rc_timezone()
    ix = int(x)
    dt = datetime.datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24 * remainder, 1)
    minute, remainder = divmod(60 * remainder, 1)
    second, remainder = divmod(60 * remainder, 1)
    microsecond = int(1e6 * remainder)
    if microsecond < 10:
        microsecond = 0  # compensate for rounding errors
    dt = datetime.datetime(
        dt.year, dt.month, dt.day, int(hour), int(minute), int(second),
        microsecond, tzinfo=UTC).astimezone(tz)

    if microsecond > 999990:  # compensate for rounding errors
        dt += datetime.timedelta(microseconds=1e6 - microsecond)

    return dt


# a version of _from_ordinalf that can operate on numpy arrays
_from_ordinalf_np_vectorized = np.vectorize(_from_ordinalf)


class strpdate2num:
    """
    Use this class to parse date strings to matplotlib datenums when
    you know the date format string of the date you are parsing.  See
    :file:`examples/load_demo.py`.
    """
    def __init__(self, fmt):
        """ fmt: any valid strptime format is supported """
        self.fmt = fmt

    def __call__(self, s):
        """s : string to be converted
           return value: a date2num float
        """
        return date2num(datetime.datetime(*time.strptime(s, self.fmt)[:6]))


# a version of dateutil.parser.parse that can operate on nump0y arrays
_dateutil_parser_parse_np_vectorized = np.vectorize(dateutil.parser.parse)


def datestr2num(d, default=None):
    """
    Convert a date string to a datenum using
    :func:`dateutil.parser.parse`.

    Parameters
    ----------
    d : string or sequence of strings
        The dates to convert.

    default : datetime instance
        The default date to use when fields are missing in `d`.
    """
    if cbook.is_string_like(d):
        dt = dateutil.parser.parse(d, default=default)
        return date2num(dt)
    else:
        if default is not None:
            d = [dateutil.parser.parse(s, default=default) for s in d]
        d = np.asarray(d)
        if not d.size:
            return d
        return date2num(_dateutil_parser_parse_np_vectorized(d))


def date2num(d):
    """
    *d* is either a :class:`datetime` instance or a sequence of datetimes.

    Return value is a floating point number (or sequence of floats)
    which gives the number of days (fraction part represents hours,
    minutes, seconds) since 0001-01-01 00:00:00 UTC, *plus* *one*.
    The addition of one here is a historical artifact.  Also, note
    that the Gregorian calendar is assumed; this is not universal
    practice.  For details, see the module docstring.
    """
    if not cbook.iterable(d):
        return _to_ordinalf(d)
    else:
        d = np.asarray(d)
        if not d.size:
            return d
        return _to_ordinalf_np_vectorized(d)


def julian2num(j):
    'Convert a Julian date (or sequence) to a matplotlib date (or sequence).'
    if cbook.iterable(j):
        j = np.asarray(j)
    return j - 1721424.5


def num2julian(n):
    'Convert a matplotlib date (or sequence) to a Julian date (or sequence).'
    if cbook.iterable(n):
        n = np.asarray(n)
    return n + 1721424.5


def num2date(x, tz=None):
    """
    *x* is a float value which gives the number of days
    (fraction part represents hours, minutes, seconds) since
    0001-01-01 00:00:00 UTC *plus* *one*.
    The addition of one here is a historical artifact.  Also, note
    that the Gregorian calendar is assumed; this is not universal
    practice.  For details, see the module docstring.

    Return value is a :class:`datetime` instance in timezone *tz* (default to
    rcparams TZ value).

    If *x* is a sequence, a sequence of :class:`datetime` objects will
    be returned.
    """
    if tz is None:
        tz = _get_rc_timezone()
    if not cbook.iterable(x):
        return _from_ordinalf(x, tz)
    else:
        x = np.asarray(x)
        if not x.size:
            return x
        return _from_ordinalf_np_vectorized(x, tz).tolist()


def drange(dstart, dend, delta):
    """
    Return a date range as float Gregorian ordinals.  *dstart* and
    *dend* are :class:`datetime` instances.  *delta* is a
    :class:`datetime.timedelta` instance.
    """
    step = (delta.days + delta.seconds / SECONDS_PER_DAY +
            delta.microseconds / MUSECONDS_PER_DAY)
    f1 = _to_ordinalf(dstart)
    f2 = _to_ordinalf(dend)

    # calculate the difference between dend and dstart in times of delta
    num = int(np.ceil((f2 - f1) / step))

    # calculate end of the interval which will be generated
    dinterval_end = dstart + num * delta

    # ensure, that an half open interval will be generated [dstart, dend)
    if dinterval_end >= dend:
        # if the endpoint is greated than dend, just subtract one delta
        dinterval_end -= delta
        num -= 1

    f2 = _to_ordinalf(dinterval_end)  # new float-endpoint
    return np.linspace(f1, f2, num + 1)

### date tickers and formatters ###


class DateFormatter(ticker.Formatter):
    """
    Tick location is seconds since the epoch.  Use a :func:`strftime`
    format string.

    Python only supports :mod:`datetime` :func:`strftime` formatting
    for years greater than 1900.  Thanks to Andrew Dalke, Dalke
    Scientific Software who contributed the :func:`strftime` code
    below to include dates earlier than this year.
    """

    illegal_s = re.compile(r"((^|[^%])(%%)*%s)")

    def __init__(self, fmt, tz=None):
        """
        *fmt* is an :func:`strftime` format string; *tz* is the
         :class:`tzinfo` instance.
        """
        if tz is None:
            tz = _get_rc_timezone()
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
        if x == 0:
            raise ValueError('DateFormatter found a value of x=0, which is '
                             'an illegal date.  This usually occurs because '
                             'you have not informed the axis that it is '
                             'plotting dates, e.g., with ax.xaxis_date()')
        dt = num2date(x, self.tz)
        return self.strftime(dt, self.fmt)

    def set_tzinfo(self, tz):
        self.tz = tz

    def _findall(self, text, substr):
        # Also finds overlaps
        sites = []
        i = 0
        while 1:
            j = text.find(substr, i)
            if j == -1:
                break
            sites.append(j)
            i = j + 1
        return sites

    # Dalke: I hope I did this math right.  Every 28 years the
    # calendar repeats, except through century leap years excepting
    # the 400 year leap years.  But only if you're using the Gregorian
    # calendar.

    def strftime(self, dt, fmt):
        fmt = self.illegal_s.sub(r"\1", fmt)
        fmt = fmt.replace("%s", "s")
        if dt.year > 1900:
            return cbook.unicode_safe(dt.strftime(fmt))

        year = dt.year
        # For every non-leap year century, advance by
        # 6 years to get into the 28-year repeat cycle
        delta = 2000 - year
        off = 6 * (delta // 100 + delta // 400)
        year = year + off

        # Move to around the year 2000
        year = year + ((2000 - year) // 28) * 28
        timetuple = dt.timetuple()
        s1 = time.strftime(fmt, (year,) + timetuple[1:])
        sites1 = self._findall(s1, str(year))

        s2 = time.strftime(fmt, (year + 28,) + timetuple[1:])
        sites2 = self._findall(s2, str(year + 28))

        sites = []
        for site in sites1:
            if site in sites2:
                sites.append(site)

        s = s1
        syear = "%4d" % (dt.year,)
        for site in sites:
            s = s[:site] + syear + s[site + 4:]

        return cbook.unicode_safe(s)


class IndexDateFormatter(ticker.Formatter):
    """
    Use with :class:`~matplotlib.ticker.IndexLocator` to cycle format
    strings by index.
    """
    def __init__(self, t, fmt, tz=None):
        """
        *t* is a sequence of dates (floating point days).  *fmt* is a
        :func:`strftime` format string.
        """
        if tz is None:
            tz = _get_rc_timezone()
        self.t = t
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
        'Return the label for time *x* at position *pos*'
        ind = int(round(x))
        if ind >= len(self.t) or ind <= 0:
            return ''

        dt = num2date(self.t[ind], self.tz)

        return cbook.unicode_safe(dt.strftime(self.fmt))


class AutoDateFormatter(ticker.Formatter):
    """
    This class attempts to figure out the best format to use.  This is
    most useful when used with the :class:`AutoDateLocator`.


    The AutoDateFormatter has a scale dictionary that maps the scale
    of the tick (the distance in days between one major tick) and a
    format string.  The default looks like this::

        self.scaled = {
           365.0  : '%Y',
           30.    : '%b %Y',
           1.0    : '%b %d %Y',
           1./24. : '%H:%M:%S',
           1. / (24. * 60.): '%H:%M:%S.%f',
           }


    The algorithm picks the key in the dictionary that is >= the
    current scale and uses that format string.  You can customize this
    dictionary by doing::


    >>> formatter = AutoDateFormatter()
    >>> formatter.scaled[1/(24.*60.)] = '%M:%S' # only show min and sec

    A custom :class:`~matplotlib.ticker.FuncFormatter` can also be used.
    The following example shows how to use a custom format function to strip
    trailing zeros from decimal seconds and adds the date to the first
    ticklabel::

        >>> def my_format_function(x, pos=None):
        ...     x = matplotlib.dates.num2date(x)
        ...     if pos == 0:
        ...         fmt = '%D %H:%M:%S.%f'
        ...     else:
        ...         fmt = '%H:%M:%S.%f'
        ...     label = x.strftime(fmt)
        ...     label = label.rstrip("0")
        ...     label = label.rstrip(".")
        ...     return label
        >>> from matplotlib.ticker import FuncFormatter
        >>> formatter.scaled[1/(24.*60.)] = FuncFormatter(my_format_function)
    """

    # This can be improved by providing some user-level direction on
    # how to choose the best format (precedence, etc...)

    # Perhaps a 'struct' that has a field for each time-type where a
    # zero would indicate "don't show" and a number would indicate
    # "show" with some sort of priority.  Same priorities could mean
    # show all with the same priority.

    # Or more simply, perhaps just a format string for each
    # possibility...

    def __init__(self, locator, tz=None, defaultfmt='%Y-%m-%d'):
        """
        Autoformat the date labels.  The default format is the one to use
        if none of the values in ``self.scaled`` are greater than the unit
        returned by ``locator._get_unit()``.
        """
        self._locator = locator
        self._tz = tz
        self.defaultfmt = defaultfmt
        self._formatter = DateFormatter(self.defaultfmt, tz)
        self.scaled = {365.0: '%Y',
                       30.: '%b %Y',
                       1.0: '%b %d %Y',
                       1. / 24.: '%H:%M:%S',
                       1. / (24. * 60.): '%H:%M:%S.%f'}

    def __call__(self, x, pos=None):
        locator_unit_scale = float(self._locator._get_unit())
        fmt = self.defaultfmt

        # Pick the first scale which is greater than the locator unit.
        for possible_scale in sorted(self.scaled):
            if possible_scale >= locator_unit_scale:
                fmt = self.scaled[possible_scale]
                break

        if isinstance(fmt, six.string_types):
            self._formatter = DateFormatter(fmt, self._tz)
            result = self._formatter(x, pos)
        elif six.callable(fmt):
            result = fmt(x, pos)
        else:
            raise TypeError('Unexpected type passed to {!r}.'.formatter(self))

        return result


class rrulewrapper:

    def __init__(self, freq, **kwargs):
        self._construct = kwargs.copy()
        self._construct["freq"] = freq
        self._rrule = rrule(**self._construct)

    def set(self, **kwargs):
        self._construct.update(kwargs)
        self._rrule = rrule(**self._construct)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._rrule, name)


class DateLocator(ticker.Locator):
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}

    def __init__(self, tz=None):
        """
        *tz* is a :class:`tzinfo` instance.
        """
        if tz is None:
            tz = _get_rc_timezone()
        self.tz = tz

    def set_tzinfo(self, tz):
        self.tz = tz

    def datalim_to_dt(self):
        dmin, dmax = self.axis.get_data_interval()
        return num2date(dmin, self.tz), num2date(dmax, self.tz)

    def viewlim_to_dt(self):
        vmin, vmax = self.axis.get_view_interval()
        return num2date(vmin, self.tz), num2date(vmax, self.tz)

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return 1

    def nonsingular(self, vmin, vmax):
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).

        """
        unit = self._get_unit()
        interval = self._get_interval()
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval
        return vmin, vmax


class RRuleLocator(DateLocator):
    # use the dateutil rrule instance

    def __init__(self, o, tz=None):
        DateLocator.__init__(self, tz)
        self.rule = o

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        if dmin > dmax:
            dmax, dmin = dmin, dmax
        delta = relativedelta(dmax, dmin)

        # We need to cap at the endpoints of valid datetime
        try:
            start = dmin - delta
        except ValueError:
            start = _from_ordinalf(1.0)

        try:
            stop = dmax + delta
        except ValueError:
            # The magic number!
            stop = _from_ordinalf(3652059.9999999)

        self.rule.set(dtstart=start, until=stop, count=self.MAXTICKS + 1)

        # estimate the number of ticks very approximately so we don't
        # have to do a very expensive (and potentially near infinite)
        # 'between' calculation, only to find out it will fail.
        nmax, nmin = date2num((dmax, dmin))
        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())
        # This estimate is only an estimate, so be really conservative
        # about bailing...
        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(
                'RRuleLocator estimated to generate %d ticks from %s to %s: '
                'exceeds Locator.MAXTICKS * 2 (%d) ' % (estimate, dmin, dmax,
                                                        self.MAXTICKS * 2))

        dates = self.rule.between(dmin, dmax, True)
        if len(dates) == 0:
            return date2num([dmin, dmax])
        return self.raise_if_exceeds(date2num(dates))

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        freq = self.rule._rrule._freq
        return self.get_unit_generic(freq)

    @staticmethod
    def get_unit_generic(freq):
        if (freq == YEARLY):
            return 365.0
        elif (freq == MONTHLY):
            return 30.0
        elif (freq == WEEKLY):
            return 7.0
        elif (freq == DAILY):
            return 1.0
        elif (freq == HOURLY):
            return (1.0 / 24.0)
        elif (freq == MINUTELY):
            return (1.0 / (24 * 60))
        elif (freq == SECONDLY):
            return (1.0 / (24 * 3600))
        else:
            # error
            return -1   # or should this just return '1'?

    def _get_interval(self):
        return self.rule._rrule._interval

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        dmin, dmax = self.datalim_to_dt()
        if dmin > dmax:
            dmax, dmin = dmin, dmax

        delta = relativedelta(dmax, dmin)

        # We need to cap at the endpoints of valid datetime
        try:
            start = dmin - delta
        except ValueError:
            start = _from_ordinalf(1.0)

        try:
            stop = dmax + delta
        except ValueError:
            # The magic number!
            stop = _from_ordinalf(3652059.9999999)

        self.rule.set(dtstart=start, until=stop)
        dmin, dmax = self.datalim_to_dt()

        vmin = self.rule.before(dmin, True)
        if not vmin:
            vmin = dmin

        vmax = self.rule.after(dmax, True)
        if not vmax:
            vmax = dmax

        vmin = date2num(vmin)
        vmax = date2num(vmax)

        return self.nonsingular(vmin, vmax)


class AutoDateLocator(DateLocator):
    """
    On autoscale, this class picks the best
    :class:`DateLocator` to set the view limits and the tick
    locations.
    """
    def __init__(self, tz=None, minticks=5, maxticks=None,
                 interval_multiples=False):
        """
        *minticks* is the minimum number of ticks desired, which is used to
        select the type of ticking (yearly, monthly, etc.).

        *maxticks* is the maximum number of ticks desired, which controls
        any interval between ticks (ticking every other, every 3, etc.).
        For really fine-grained control, this can be a dictionary mapping
        individual rrule frequency constants (YEARLY, MONTHLY, etc.)
        to their own maximum number of ticks.  This can be used to keep
        the number of ticks appropriate to the format chosen in
        :class:`AutoDateFormatter`. Any frequency not specified in this
        dictionary is given a default value.

        *tz* is a :class:`tzinfo` instance.

        *interval_multiples* is a boolean that indicates whether ticks
        should be chosen to be multiple of the interval. This will lock
        ticks to 'nicer' locations. For example, this will force the
        ticks to be at hours 0,6,12,18 when hourly ticking is done at
        6 hour intervals.

        The AutoDateLocator has an interval dictionary that maps the
        frequency of the tick (a constant from dateutil.rrule) and a
        multiple allowed for that ticking.  The default looks like this::

          self.intervald = {
            YEARLY  : [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                      1000, 2000, 4000, 5000, 10000],
            MONTHLY : [1, 2, 3, 4, 6],
            DAILY   : [1, 2, 3, 7, 14],
            HOURLY  : [1, 2, 3, 4, 6, 12],
            MINUTELY: [1, 5, 10, 15, 30],
            SECONDLY: [1, 5, 10, 15, 30],
            MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                           5000, 10000, 20000, 50000, 100000, 200000, 500000,
                           1000000],
            }

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 7 days is sensible
        for daily ticks, but for minutes/seconds, 15 or 30 make sense.
        You can customize this dictionary by doing::

          locator = AutoDateLocator()
          locator.intervald[HOURLY] = [3] # only show every 3 hours
        """
        DateLocator.__init__(self, tz)
        self._locator = YearLocator()
        self._freq = YEARLY
        self._freqs = [YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY,
                       SECONDLY, MICROSECONDLY]
        self.minticks = minticks

        self.maxticks = {YEARLY: 11, MONTHLY: 12, DAILY: 11, HOURLY: 12,
                         MINUTELY: 11, SECONDLY: 11, MICROSECONDLY: 8}
        if maxticks is not None:
            try:
                self.maxticks.update(maxticks)
            except TypeError:
                # Assume we were given an integer. Use this as the maximum
                # number of ticks for every frequency and create a
                # dictionary for this
                self.maxticks = dict(zip(self._freqs,
                                         [maxticks] * len(self._freqs)))
        self.interval_multiples = interval_multiples
        self.intervald = {
            YEARLY:   [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                       1000, 2000, 4000, 5000, 10000],
            MONTHLY:  [1, 2, 3, 4, 6],
            DAILY:    [1, 2, 3, 7, 14, 21],
            HOURLY:   [1, 2, 3, 4, 6, 12],
            MINUTELY: [1, 5, 10, 15, 30],
            SECONDLY: [1, 5, 10, 15, 30],
            MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                            5000, 10000, 20000, 50000, 100000, 200000, 500000,
                            1000000]}
        self._byranges = [None, list(xrange(1, 13)), list(xrange(1, 32)),
                          list(xrange(0, 24)), list(xrange(0, 60)),
                          list(xrange(0, 60)), None]

    def __call__(self):
        'Return the locations of the ticks'
        self.refresh()
        return self._locator()

    def nonsingular(self, vmin, vmax):
        # whatever is thrown at us, we can scale the unit.
        # But default nonsingular date plots at an ~4 year period.
        if vmin == vmax:
            vmin = vmin - 365 * 2
            vmax = vmax + 365 * 2
        return vmin, vmax

    def set_axis(self, axis):
        DateLocator.set_axis(self, axis)
        self._locator.set_axis(axis)

    def refresh(self):
        'Refresh internal information based on current limits.'
        dmin, dmax = self.viewlim_to_dt()
        self._locator = self.get_locator(dmin, dmax)

    def _get_unit(self):
        if self._freq in [MICROSECONDLY]:
            return 1. / MUSECONDS_PER_DAY
        else:
            return RRuleLocator.get_unit_generic(self._freq)

    def autoscale(self):
        'Try to choose the view limits intelligently.'
        dmin, dmax = self.datalim_to_dt()
        self._locator = self.get_locator(dmin, dmax)
        return self._locator.autoscale()

    def get_locator(self, dmin, dmax):
        'Pick the best locator based on a distance.'
        delta = relativedelta(dmax, dmin)

        # take absolute difference
        if dmin > dmax:
            delta = -delta

        numYears = (delta.years * 1.0)
        numMonths = (numYears * 12.0) + delta.months
        numDays = (numMonths * 31.0) + delta.days
        numHours = (numDays * 24.0) + delta.hours
        numMinutes = (numHours * 60.0) + delta.minutes
        numSeconds = (numMinutes * 60.0) + delta.seconds
        numMicroseconds = (numSeconds * 1e6) + delta.microseconds

        nums = [numYears, numMonths, numDays, numHours, numMinutes,
                numSeconds, numMicroseconds]

        use_rrule_locator = [True] * 6 + [False]

        # Default setting of bymonth, etc. to pass to rrule
        # [unused (for year), bymonth, bymonthday, byhour, byminute,
        #  bysecond, unused (for microseconds)]
        byranges = [None, 1, 1, 0, 0, 0, None]

        # Loop over all the frequencies and try to find one that gives at
        # least a minticks tick positions.  Once this is found, look for
        # an interval from an list specific to that frequency that gives no
        # more than maxticks tick positions. Also, set up some ranges
        # (bymonth, etc.) as appropriate to be passed to rrulewrapper.
        for i, (freq, num) in enumerate(zip(self._freqs, nums)):
            # If this particular frequency doesn't give enough ticks, continue
            if num < self.minticks:
                # Since we're not using this particular frequency, set
                # the corresponding by_ to None so the rrule can act as
                # appropriate
                byranges[i] = None
                continue

            # Find the first available interval that doesn't give too many
            # ticks
            for interval in self.intervald[freq]:
                if num <= interval * (self.maxticks[freq] - 1):
                    break
            else:
                # We went through the whole loop without breaking, default to
                # the last interval in the list and raise a warning
                warnings.warn('AutoDateLocator was unable to pick an '
                              'appropriate interval for this date range. '
                              'It may be necessary to add an interval value '
                              "to the AutoDateLocator's intervald dictionary."
                              ' Defaulting to {0}.'.format(interval))

            # Set some parameters as appropriate
            self._freq = freq

            if self._byranges[i] and self.interval_multiples:
                byranges[i] = self._byranges[i][::interval]
                interval = 1
            else:
                byranges[i] = self._byranges[i]

            # We found what frequency to use
            break
        else:
            raise ValueError('No sensible date limit could be found in the '
                             'AutoDateLocator.')

        if use_rrule_locator[i]:
            _, bymonth, bymonthday, byhour, byminute, bysecond, _ = byranges

            rrule = rrulewrapper(self._freq, interval=interval,
                                 dtstart=dmin, until=dmax,
                                 bymonth=bymonth, bymonthday=bymonthday,
                                 byhour=byhour, byminute=byminute,
                                 bysecond=bysecond)

            locator = RRuleLocator(rrule, self.tz)
        else:
            locator = MicrosecondLocator(interval, tz=self.tz)

        locator.set_axis(self.axis)

        locator.set_view_interval(*self.axis.get_view_interval())
        locator.set_data_interval(*self.axis.get_data_interval())
        return locator


class YearLocator(DateLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples::

      # Tick every year on Jan 1st
      locator = YearLocator()

      # Tick every 5 years on July 4th
      locator = YearLocator(5, month=7, day=4)
    """
    def __init__(self, base=1, month=1, day=1, tz=None):
        """
        Mark years that are multiple of base on a given month and day
        (default jan 1).
        """
        DateLocator.__init__(self, tz)
        self.base = ticker.Base(base)
        self.replaced = {'month':  month,
                         'day':    day,
                         'hour':   0,
                         'minute': 0,
                         'second': 0,
                         'tzinfo': tz
                         }

    def __call__(self):
        dmin, dmax = self.viewlim_to_dt()
        ymin = self.base.le(dmin.year)
        ymax = self.base.ge(dmax.year)

        ticks = [dmin.replace(year=ymin, **self.replaced)]
        while 1:
            dt = ticks[-1]
            if dt.year >= ymax:
                return date2num(ticks)
            year = dt.year + self.base.get_base()
            ticks.append(dt.replace(year=year, **self.replaced))

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        dmin, dmax = self.datalim_to_dt()

        ymin = self.base.le(dmin.year)
        ymax = self.base.ge(dmax.year)
        vmin = dmin.replace(year=ymin, **self.replaced)
        vmax = dmax.replace(year=ymax, **self.replaced)

        vmin = date2num(vmin)
        vmax = date2num(vmax)
        return self.nonsingular(vmin, vmax)


class MonthLocator(RRuleLocator):
    """
    Make ticks on occurances of each month month, e.g., 1, 3, 12.
    """
    def __init__(self,  bymonth=None, bymonthday=1, interval=1, tz=None):
        """
        Mark every month in *bymonth*; *bymonth* can be an int or
        sequence.  Default is ``range(1,13)``, i.e. every month.

        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second occurance.
        """
        if bymonth is None:
            bymonth = list(xrange(1, 13))
        o = rrulewrapper(MONTHLY, bymonth=bymonth, bymonthday=bymonthday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)


class WeekdayLocator(RRuleLocator):
    """
    Make ticks on occurances of each weekday.
    """

    def __init__(self,  byweekday=1, interval=1, tz=None):
        """
        Mark every weekday in *byweekday*; *byweekday* can be a number or
        sequence.

        Elements of *byweekday* must be one of MO, TU, WE, TH, FR, SA,
        SU, the constants from :mod:`dateutil.rrule`, which have been
        imported into the :mod:`matplotlib.dates` namespace.

        *interval* specifies the number of weeks to skip.  For example,
        ``interval=2`` plots every second week.
        """
        o = rrulewrapper(DAILY, byweekday=byweekday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)


class DayLocator(RRuleLocator):
    """
    Make ticks on occurances of each day of the month.  For example,
    1, 15, 30.
    """
    def __init__(self,  bymonthday=None, interval=1, tz=None):
        """
        Mark every day in *bymonthday*; *bymonthday* can be an int or
        sequence.

        Default is to tick every day of the month: ``bymonthday=range(1,32)``
        """
        if bymonthday is None:
            bymonthday = list(xrange(1, 32))
        o = rrulewrapper(DAILY, bymonthday=bymonthday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)


class HourLocator(RRuleLocator):
    """
    Make ticks on occurances of each hour.
    """
    def __init__(self,  byhour=None, interval=1, tz=None):
        """
        Mark every hour in *byhour*; *byhour* can be an int or sequence.
        Default is to tick every hour: ``byhour=range(24)``

        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second occurrence.
        """
        if byhour is None:
            byhour = list(xrange(24))
        rule = rrulewrapper(HOURLY, byhour=byhour, interval=interval,
                            byminute=0, bysecond=0)
        RRuleLocator.__init__(self, rule, tz)


class MinuteLocator(RRuleLocator):
    """
    Make ticks on occurances of each minute.
    """
    def __init__(self,  byminute=None, interval=1, tz=None):
        """
        Mark every minute in *byminute*; *byminute* can be an int or
        sequence.  Default is to tick every minute: ``byminute=range(60)``

        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second occurrence.
        """
        if byminute is None:
            byminute = list(xrange(60))
        rule = rrulewrapper(MINUTELY, byminute=byminute, interval=interval,
                            bysecond=0)
        RRuleLocator.__init__(self, rule, tz)


class SecondLocator(RRuleLocator):
    """
    Make ticks on occurances of each second.
    """
    def __init__(self,  bysecond=None, interval=1, tz=None):
        """
        Mark every second in *bysecond*; *bysecond* can be an int or
        sequence.  Default is to tick every second: ``bysecond = range(60)``

        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second occurrence.

        """
        if bysecond is None:
            bysecond = list(xrange(60))
        rule = rrulewrapper(SECONDLY, bysecond=bysecond, interval=interval)
        RRuleLocator.__init__(self, rule, tz)


class MicrosecondLocator(DateLocator):
    """
    Make ticks on occurances of each microsecond.

    """
    def __init__(self, interval=1, tz=None):
        """
        *interval* is the interval between each iteration.  For
        example, if ``interval=2``, mark every second microsecond.

        """
        self._interval = interval
        self._wrapped_locator = ticker.MultipleLocator(interval)
        self.tz = tz

    def set_axis(self, axis):
        self._wrapped_locator.set_axis(axis)
        return DateLocator.set_axis(self, axis)

    def set_view_interval(self, vmin, vmax):
        self._wrapped_locator.set_view_interval(vmin, vmax)
        return DateLocator.set_view_interval(self, vmin, vmax)

    def set_data_interval(self, vmin, vmax):
        self._wrapped_locator.set_data_interval(vmin, vmax)
        return DateLocator.set_data_interval(self, vmin, vmax)

    def __call__(self, *args, **kwargs):
        vmin, vmax = self.axis.get_view_interval()
        vmin *= MUSECONDS_PER_DAY
        vmax *= MUSECONDS_PER_DAY
        ticks = self._wrapped_locator.tick_values(vmin, vmax)
        ticks = [tick / MUSECONDS_PER_DAY for tick in ticks]
        return ticks

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1. / MUSECONDS_PER_DAY

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return self._interval


def _close_to_dt(d1, d2, epsilon=5):
    'Assert that datetimes *d1* and *d2* are within *epsilon* microseconds.'
    delta = d2 - d1
    mus = abs(delta.days * MUSECONDS_PER_DAY + delta.seconds * 1e6 +
              delta.microseconds)
    assert(mus < epsilon)


def _close_to_num(o1, o2, epsilon=5):
    """
    Assert that float ordinals *o1* and *o2* are within *epsilon*
    microseconds.
    """
    delta = abs((o2 - o1) * MUSECONDS_PER_DAY)
    assert(delta < epsilon)


def epoch2num(e):
    """
    Convert an epoch or sequence of epochs to the new date format,
    that is days since 0001.
    """
    spd = 24. * 3600.
    return 719163 + np.asarray(e) / spd


def num2epoch(d):
    """
    Convert days since 0001 to epoch.  *d* can be a number or sequence.
    """
    spd = 24. * 3600.
    return (np.asarray(d) - 719163) * spd


def mx2num(mxdates):
    """
    Convert mx :class:`datetime` instance (or sequence of mx
    instances) to the new date format.
    """
    scalar = False
    if not cbook.iterable(mxdates):
        scalar = True
        mxdates = [mxdates]
    ret = epoch2num([m.ticks() for m in mxdates])
    if scalar:
        return ret[0]
    else:
        return ret


def date_ticker_factory(span, tz=None, numticks=5):
    """
    Create a date locator with *numticks* (approx) and a date formatter
    for *span* in days.  Return value is (locator, formatter).
    """

    if span == 0:
        span = 1 / 24.

    minutes = span * 24 * 60
    hours = span * 24
    days = span
    weeks = span / 7.
    months = span / 31.  # approx
    years = span / 365.

    if years > numticks:
        locator = YearLocator(int(years / numticks), tz=tz)  # define
        fmt = '%Y'
    elif months > numticks:
        locator = MonthLocator(tz=tz)
        fmt = '%b %Y'
    elif weeks > numticks:
        locator = WeekdayLocator(tz=tz)
        fmt = '%a, %b %d'
    elif days > numticks:
        locator = DayLocator(interval=int(math.ceil(days / numticks)), tz=tz)
        fmt = '%b %d'
    elif hours > numticks:
        locator = HourLocator(interval=int(math.ceil(hours / numticks)), tz=tz)
        fmt = '%H:%M\n%b %d'
    elif minutes > numticks:
        locator = MinuteLocator(interval=int(math.ceil(minutes / numticks)),
                                tz=tz)
        fmt = '%H:%M:%S'
    else:
        locator = MinuteLocator(tz=tz)
        fmt = '%H:%M:%S'

    formatter = DateFormatter(fmt, tz=tz)
    return locator, formatter


def seconds(s):
    'Return seconds as days.'
    return float(s) / SEC_PER_DAY


def minutes(m):
    'Return minutes as days.'
    return float(m) / MINUTES_PER_DAY


def hours(h):
    'Return hours as days.'
    return h / 24.


def weeks(w):
    'Return weeks as days.'
    return w * 7.


class DateConverter(units.ConversionInterface):
    """
    Converter for datetime.date and datetime.datetime data,
    or for date/time data represented as it would be converted
    by :func:`date2num`.

    The 'unit' tag for such data is None or a tzinfo instance.
    """

    @staticmethod
    def axisinfo(unit, axis):
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        majloc = AutoDateLocator(tz=tz)
        majfmt = AutoDateFormatter(majloc, tz=tz)
        datemin = datetime.date(2000, 1, 1)
        datemax = datetime.date(2010, 1, 1)

        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))

    @staticmethod
    def convert(value, unit, axis):
        """
        If *value* is not already a number or sequence of numbers,
        convert it with :func:`date2num`.

        The *unit* and *axis* arguments are not used.
        """
        if units.ConversionInterface.is_numlike(value):
            return value
        return date2num(value)

    @staticmethod
    def default_units(x, axis):
        'Return the tzinfo instance of *x* or of its first element, or None'
        if isinstance(x, np.ndarray):
            x = x.ravel()

        try:
            x = x[0]
        except (TypeError, IndexError):
            pass

        try:
            return x.tzinfo
        except AttributeError:
            pass
        return None


units.registry[datetime.date] = DateConverter()
units.registry[datetime.datetime] = DateConverter()
