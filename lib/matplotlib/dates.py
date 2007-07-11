#!/usr/bin/env python
"""

Matplotlib provides sophisticated date plotting capabilities, standing
on the shoulders of python datetime, the add-on modules pytz and
dateutils.  datetime objects are converted to floating point numbers
which represent the number of days since 0001-01-01 UTC.  The helper
functions date2num, num2date and drange are used to facilitate easy
conversion to and from datetime and numeric ranges.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See matplotlib.tickers for
general information on tick locators and formatters.  These are
described below.

All the matplotlib date converters, tickers and formatters are
timezone aware, and the default timezone is given by the timezone
parameter in your matplotlibrc file.  If you leave out a tz timezone
instance, the default from your rc file will be assumed.  If you want
to use a custom time zone, pass a pytz.timezone instance
with the tz keyword argument to num2date, plot_date, and any custom
date tickers or locators you create.  See http://pytz.sourceforge.net
for information on pytz and timezone handling.

The dateutil module (http://labix.org/python-dateutil)
provides additional code to handle
date ticking, making it easy to place ticks on any kinds of dates -
see examples below.

Date tickers -

  Most of the date tickers can locate single or multiple values.  Eg

    # tick on mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on mondays and saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

  In addition, most of the constructors take an interval argument.

    # tick on mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

  The rrule locator allows completely general date ticking

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

  Here are all the date tickers

    * MinuteLocator  - locate minutes

    * HourLocator    - locate hours

    * DayLocator     - locate specifed days of the month

    * WeekdayLocator - Locate days of the week, eg MO, TU

    * MonthLocator   - locate months, eg 7 for july

    * YearLocator    - locate years that are multiples of base

    * RRuleLocator - locate using a matplotlib.dates.rrulewrapper.
        The rrulewrapper is a simple wrapper around a dateutils.rrule
        https://moin.conectiva.com.br/DateUtil which allow almost
        arbitrary date tick specifications.  See
        examples/date_demo_rrule.py


Date formatters

  DateFormatter - use strftime format strings

  DateIndexFormatter - date plots with implicit x indexing.

"""
import sys, re, time, math, datetime
import locale

import pytz
import matplotlib
import numpy as npy

import matplotlib.units as units
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker

from pytz import timezone
from dateutil.rrule import rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, \
     MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY
from dateutil.relativedelta import relativedelta
import dateutil.parser


__all__ = ( 'date2num', 'num2date', 'drange', 'epoch2num',
            'num2epoch', 'mx2num', 'DateFormatter',
            'IndexDateFormatter', 'DateLocator', 'RRuleLocator',
            'YearLocator', 'MonthLocator', 'WeekdayLocator',
            'DayLocator', 'HourLocator', 'MinuteLocator',
            'SecondLocator', 'rrule', 'MO', 'TU', 'WE', 'TH', 'FR',
            'SA', 'SU', 'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY',
            'HOURLY', 'MINUTELY', 'SECONDLY', 'relativedelta',
            'seconds', 'minutes', 'hours', 'weeks')



UTC = pytz.timezone('UTC')

def _get_rc_timezone():
    s = matplotlib.rcParams['timezone']
    return pytz.timezone(s)


HOURS_PER_DAY = 24.
MINUTES_PER_DAY  = 60.*HOURS_PER_DAY
SECONDS_PER_DAY =  60.*MINUTES_PER_DAY
MUSECONDS_PER_DAY = 1e6*SECONDS_PER_DAY
SEC_PER_MIN = 60
SEC_PER_HOUR = 3600
SEC_PER_DAY = SEC_PER_HOUR * 24
SEC_PER_WEEK = SEC_PER_DAY * 7
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = (
    MO, TU, WE, TH, FR, SA, SU)
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)




def _to_ordinalf(dt):
    """
    convert datetime to the Gregorian date as UTC float days,
    preserving hours, minutes, seconds and microseconds.  return value
    is a float
    """

    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta

    base =  float(dt.toordinal())
    if hasattr(dt, 'hour'):
        base += (dt.hour/HOURS_PER_DAY + dt.minute/MINUTES_PER_DAY +
                 dt.second/SECONDS_PER_DAY + dt.microsecond/MUSECONDS_PER_DAY
                 )
    return base

def _from_ordinalf(x, tz=None):
    """
    convert Gregorian float of the date, preserving hours, minutes,
    seconds and microseconds.  return value is a datetime
    """
    if tz is None: tz = _get_rc_timezone()
    ix = int(x)
    dt = datetime.datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24*remainder, 1)
    minute, remainder = divmod(60*remainder, 1)
    second, remainder = divmod(60*remainder, 1)
    microsecond = int(1e6*remainder)
    if microsecond<10: microsecond=0 # compensate for rounding errors
    dt = datetime.datetime(
        dt.year, dt.month, dt.day, int(hour), int(minute), int(second),
        microsecond, tzinfo=UTC).astimezone(tz)

    if microsecond>999990:  # compensate for rounding errors
        dt += datetime.timedelta(microseconds=1e6-microsecond)

    return dt

class strpdate2num:
    """
    Use this class to parse date strings to matplotlib datenums when
    you know the date format string of the date you are parsing.  See
    examples/load_demo.py
    """
    def __init__(self, fmt):
        """ fmt: any valid strptime format is supported """
        self.fmt = fmt

    def __call__(self, s):
        """s : string to be converted
           return value: a date2num float
        """
        return date2num(datetime.datetime(*time.strptime(s, self.fmt)[:6]))

def datestr2num(d):
    """
    Convert a date string to a datenum using dateutil.parser.parse
    d can be a single string or a sequence of strings
    """
    if cbook.is_string_like(d):
        dt = dateutil.parser.parse(d)
        return date2num(dt)
    else:
        return date2num([dateutil.parser.parse(s) for s in d])


def date2num(d):
    """
    d is either a datetime instance or a sequence of datetimes

    return value is a floating point number (or sequence of floats)
    which gives number of days (fraction part represents hours,
    minutes, seconds) since 0001-01-01 00:00:00 UTC
    """
    if not cbook.iterable(d): return _to_ordinalf(d)
    else: return npy.asarray([_to_ordinalf(val) for val in d])


def julian2num(j):
    'convert a Julian date (or sequence) to a matplotlib date (or sequence)'
    if cbook.iterable(j): j = npy.asarray(j)
    return j + 1721425.5

def num2julian(n):
    'convert a matplotlib date (or seguence) to a Julian date (or sequence)'
    if cbook.iterable(n): n = npy.asarray(n)
    return n - 1721425.5

def num2date(x, tz=None):
    """
    x is a float value which gives number of days (fraction part
    represents hours, minutes, seconds) since 0001-01-01 00:00:00 UTC

    Return value is a datetime instance in timezone tz (default to
    rcparams TZ value)

    if x is a sequence, a sequence of datetimes will be returned
    """
    if tz is None: tz = _get_rc_timezone()
    if not cbook.iterable(x): return _from_ordinalf(x, tz)
    else: return [_from_ordinalf(val, tz) for val in x]

def drange(dstart, dend, delta):
    """
    Return a date range as float gregorian ordinals.  dstart and dend
    are datetime instances.  delta is a datetime.timedelta instance
    """
    step = (delta.days + delta.seconds/SECONDS_PER_DAY +
            delta.microseconds/MUSECONDS_PER_DAY)
    f1 = _to_ordinalf(dstart)
    f2 = _to_ordinalf(dend)
    return npy.arange(f1, f2, step)



### date tickers and formatters ###



class DateFormatter(ticker.Formatter):
    """
    Tick location is seconds since the epoch.  Use a strftime format
    string

    python only supports datetime strftime formatting for years
    greater than 1900.  Thanks to Andrew Dalke, Dalke Scientific
    Software who contributed the strftime code below to include dates
    earlier than this year
    """

    illegal_s = re.compile(r"((^|[^%])(%%)*%s)")

    def __init__(self, fmt, tz=None):
        """
        fmt is an strftime format string; tz is the tzinfo instance
        """
        if tz is None: tz = _get_rc_timezone()
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
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
            i=j+1
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
        off = 6*(delta // 100 + delta // 400)
        year = year + off

        # Move to around the year 2000
        year = year + ((2000 - year)//28)*28
        timetuple = dt.timetuple()
        s1 = time.strftime(fmt, (year,) + timetuple[1:])
        sites1 = self._findall(s1, str(year))

        s2 = time.strftime(fmt, (year+28,) + timetuple[1:])
        sites2 = self._findall(s2, str(year+28))

        sites = []
        for site in sites1:
            if site in sites2:
                sites.append(site)

        s = s1
        syear = "%4d" % (dt.year,)
        for site in sites:
            s = s[:site] + syear + s[site+4:]

        return cbook.unicode_safe(s)



class IndexDateFormatter(ticker.Formatter):
    """
    Use with IndexLocator to cycle format strings by index.
    """
    def __init__(self, t, fmt, tz=None):
        """
        t is a sequence of dates floating point days).  fmt is a
        strftime format string

        """
        if tz is None: tz = _get_rc_timezone()
        self.t = t
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(round(x))
        if ind>=len(self.t) or ind<=0: return ''

        dt = num2date(self.t[ind], self.tz)

        return cbook.unicode_safe(dt.strftime(self.fmt))


class AutoDateFormatter(ticker.Formatter):
    """
    This class attempt to figure out the best format to use.  This is
    most useful when used with the AutoDateLocator.
    """

    # This can be improved by providing some user-level direction on
    # how to choose the best format (precedence, etc...)

    # Perhaps a 'struct' that has a field for each time-type where a
    # zero would indicate "don't show" and a number would indicate
    # "show" with some sort of priority.  Same priorities could mean
    # show all with the same priority.

    # Or more simply, perhaps just a format string for each
    # possibility...

    def __init__(self, locator, tz=None):
        self._locator = locator
        self._formatter = DateFormatter("%b %d %Y %H:%M:%S %Z", tz)
        self._tz = tz

    def __call__(self, x, pos=0):
        scale = float( self._locator._get_unit() )

        if ( scale == 365.0 ):
            self._formatter = DateFormatter("%Y", self._tz)
        elif ( scale == 30.0 ):
            self._formatter = DateFormatter("%b %Y", self._tz)
        elif ( (scale == 1.0) or (scale == 7.0) ):
            self._formatter = DateFormatter("%b %d %Y", self._tz)
        elif ( scale == (1.0/24.0) ):
            self._formatter = DateFormatter("%H:%M:%S %Z", self._tz)
        elif ( scale == (1.0/(24*60)) ):
            self._formatter = DateFormatter("%H:%M:%S %Z", self._tz)
        elif ( scale == (1.0/(24*3600)) ):
            self._formatter = DateFormatter("%H:%M:%S %Z", self._tz)
        else:
            self._formatter = DateFormatter("%b %d %Y %H:%M:%S %Z", self._tz)

        return self._formatter(x, pos)


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
    hms0d = {'byhour':0, 'byminute':0,'bysecond':0}
    def __init__(self, tz=None):
        """
        tz is the tzinfo instance
        """
        if tz is None: tz = _get_rc_timezone()
        self.tz = tz

    def set_tzinfo(self, tz):
        self.tz = tz

    def datalim_to_dt(self):
        self.verify_intervals()
        dmin, dmax = self.dataInterval.get_bounds()
        return num2date(dmin, self.tz), num2date(dmax, self.tz)

    def viewlim_to_dt(self):
        self.verify_intervals()
        vmin, vmax = self.viewInterval.get_bounds()
        return num2date(vmin, self.tz), num2date(vmax, self.tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 1

    def nonsingular(self, vmin, vmax):
        unit = self._get_unit()
        vmin -= 2*unit
        vmax += 2*unit
        return vmin, vmax

class RRuleLocator(DateLocator):
    # use the dateutil rrule instance

    def __init__(self, o, tz=None):
        DateLocator.__init__(self, tz)
        self.rule = o

    def __call__(self):
        self.verify_intervals()

        # if no data have been set, this will tank with a ValueError
        try: dmin, dmax = self.viewlim_to_dt()
        except ValueError: return []

        if dmin>dmax:
            dmax, dmin = dmin, dmax
        delta = relativedelta(dmax, dmin)
        self.rule.set(dtstart=dmin-delta, until=dmax+delta)
        dates = self.rule.between(dmin, dmax, True)
        return date2num(dates)

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        freq = self.rule._rrule._freq
        if ( freq == YEARLY ):
            return 365
        elif ( freq == MONTHLY ):
            return 30
        elif ( freq == WEEKLY ):
            return 7
        elif ( freq == DAILY ):
            return 1
        elif ( freq == HOURLY ):
            return (1.0/24.0)
        elif ( freq == MINUTELY ):
            return (1.0/(24*60))
        elif ( freq == SECONDLY ):
            return (1.0/(24*3600))
        else:
            # error
            return -1   #or should this just return '1'?

    def autoscale(self):
        """
        Set the view limits to include the data range
        """
        self.verify_intervals()
        dmin, dmax = self.datalim_to_dt()
        if dmin>dmax:
            dmax, dmin = dmin, dmax

        delta = relativedelta(dmax, dmin)
        self.rule.set(dtstart=dmin-delta, until=dmax+delta)
        dmin, dmax = self.datalim_to_dt()


        vmin = self.rule.before(dmin, True)
        if not vmin: vmin=dmin

        vmax = self.rule.after(dmax, True)
        if not vmax: vmax=dmax

        vmin = date2num(vmin)
        vmax = date2num(vmax)

        return self.nonsingular(vmin, vmax)


class AutoDateLocator(DateLocator):
    """
    On autoscale this class picks the best MultipleDateLocator to set the
    view limits and the tick locs.
    """
    def __init__(self, tz=None):
        DateLocator.__init__(self, tz)
        self._locator = YearLocator()
        self._freq = YEARLY

    def __call__(self):
        'Return the locations of the ticks'
        self.refresh()
        return self._locator()

    def refresh(self):
        'refresh internal information based on current lim'
        dmin, dmax = self.viewlim_to_dt()
        self._locator = self.get_locator(dmin, dmax)

    def _get_unit(self):
        if ( self._freq == YEARLY ):
            return 365.0
        elif ( self._freq == MONTHLY ):
            return 30.0
        elif ( self._freq == WEEKLY ):
            return 7.0
        elif ( self._freq == DAILY ):
            return 1.0
        elif ( self._freq == HOURLY ):
            return 1.0/24
        elif ( self._freq == MINUTELY ):
            return 1.0/(24*60)
        elif ( self._freq == SECONDLY ):
            return 1.0/(24*3600)
        else:
            # error
            return -1

    def autoscale(self):
        'Try to choose the view limits intelligently'

        self.verify_intervals()
        dmin, dmax = self.datalim_to_dt()
        self._locator = self.get_locator(dmin, dmax)
        return self._locator.autoscale()

    def get_locator(self, dmin, dmax):
        'pick the best locator based on a distance'

        delta = relativedelta(dmax, dmin)

        numYears = (delta.years * 1.0)
        numMonths = (numYears * 12.0) + delta.months
        numDays = (numMonths * 31.0) + delta.days
        numHours = (numDays * 24.0) + delta.hours
        numMinutes = (numHours * 60.0) + delta.minutes
        numSeconds = (numMinutes * 60.0) + delta.seconds

        numticks = 5

        # self._freq = YEARLY
        interval = 1
        bymonth = 1
        bymonthday = 1
        byhour = 0
        byminute = 0
        bysecond = 0

        if ( numYears >= numticks ):
            self._freq = YEARLY
        elif ( numMonths >= numticks ):
            self._freq = MONTHLY
            bymonth = range(1, 13)
            if ( (0 <= numMonths) and (numMonths <= 14) ):
                interval = 1      # show every month
            elif ( (15 <= numMonths) and (numMonths <= 29) ):
                interval = 3      # show every 3 months
            elif ( (30 <= numMonths) and (numMonths <= 44) ):
                interval = 4      # show every 4 months
            else:   # 45 <= numMonths <= 59
                interval = 6      # show every 6 months
        elif ( numDays >= numticks ):
            self._freq = DAILY
            bymonth = None
            bymonthday = range(1, 32)
            if ( (0 <= numDays) and (numDays <= 9) ):
                interval = 1      # show every day
            elif ( (10 <= numDays) and (numDays <= 19) ):
                interval = 2      # show every 2 days
            elif ( (20 <= numDays) and (numDays <= 49) ):
                interval = 3      # show every 3 days
            elif ( (50 <= numDays) and (numDays <= 99) ):
                interval = 7      # show every 1 week
            else:   # 100 <= numDays <= ~150
                interval = 14     # show every 2 weeks
        elif ( numHours >= numticks ):
            self._freq = HOURLY
            bymonth = None
            bymonthday = None
            byhour = range(0, 24)      # show every hour
            if ( (0 <= numHours) and (numHours <= 14) ):
                interval = 1      # show every hour
            elif ( (15 <= numHours) and (numHours <= 30) ):
                interval = 2      # show every 2 hours
            elif ( (30 <= numHours) and (numHours <= 45) ):
                interval = 3      # show every 3 hours
            elif ( (45 <= numHours) and (numHours <= 68) ):
                interval = 4      # show every 4 hours
            elif ( (68 <= numHours) and (numHours <= 90) ):
                interval = 6      # show every 6 hours
            else:   # 90 <= numHours <= 120
                interval = 12     # show every 12 hours
        elif ( numMinutes >= numticks ):
            self._freq = MINUTELY
            bymonth = None
            bymonthday = None
            byhour = None
            byminute = range(0, 60)
            if ( numMinutes > (10.0 * numticks) ):
                interval = 10
            # end if
        elif ( numSeconds >= numticks ):
            self._freq = SECONDLY
            bymonth = None
            bymonthday = None
            byhour = None
            byminute = None
            bysecond = range(0, 60)
            if ( numSeconds > (10.0 * numticks) ):
                interval = 10
            # end if
        else:
            # do what?
            #   microseconds as floats, but floats from what reference point?
            pass


        rrule = rrulewrapper( self._freq, interval=interval,          \
                              dtstart=dmin, until=dmax,               \
                              bymonth=bymonth, bymonthday=bymonthday, \
                              byhour=byhour, byminute = byminute,     \
                              bysecond=bysecond )

        locator = RRuleLocator(rrule, self.tz)

        locator.set_view_interval(self.viewInterval)
        locator.set_data_interval(self.dataInterval)
        return locator


class YearLocator(DateLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples:
    # Tick every year on Jan 1st
    locator = YearLocator()

    # Tick every 5 years on July 4th
    locator = YearLocator(5, month=7, day=4)


    """
    def __init__(self, base=1, month=1, day=1, tz=None):
        """
        mark years that are multiple of base on a given month and day
        (default jan 1)
        """
        DateLocator.__init__(self, tz)
        self.base = ticker.Base(base)
        self.replaced = { 'month'  : month,
                          'day'    : day,
                          'hour'   : 0,
                          'minute' : 0,
                          'second' : 0,
                          'tzinfo' : tz
                          }


    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 365

    def __call__(self):
        self.verify_intervals()

        dmin, dmax = self.viewlim_to_dt()
        ymin = self.base.le(dmin.year)
        ymax = self.base.ge(dmax.year)


        ticks = [dmin.replace(year=ymin, **self.replaced)]
        while 1:
            dt = ticks[-1]
            if dt.year>=ymax: return date2num(ticks)
            year = dt.year + self.base.get_base()
            ticks.append(dt.replace(year=year, **self.replaced))

    def autoscale(self):
        """
        Set the view limits to include the data range
        """
        self.verify_intervals()
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
    Make ticks on occurances of each month month, eg 1, 3, 12
    """
    def __init__(self,  bymonth=None, bymonthday=1, interval=1, tz=None):
        """
        mark every month in bymonth; bymonth can be an int or
        sequence.  default is range(1,13), ie every month

        interval is the interval between each iteration.  Eg, if
        interval=2, mark every second occurance
        """
        if bymonth is None: bymonth=range(1,13)
        o = rrulewrapper(MONTHLY, bymonth=bymonth, bymonthday=bymonthday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 30


class WeekdayLocator(RRuleLocator):
    """
    Make ticks on occurances of each weekday
    """

    def __init__(self,  byweekday=1, interval=1, tz=None):
        """
        mark every weekday in byweekday; byweekday can be a number or
        sequence

        elements of byweekday must be one of MO, TU, WE, TH, FR, SA,
        SU, the constants from dateutils.rrule

        interval specifies the number of weeks to skip.  Ie interval=2
        plots every second week

        """
        o = rrulewrapper(DAILY, byweekday=byweekday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 7


class DayLocator(RRuleLocator):
    """
    Make ticks on occurances of each day of the month, eg 1, 15, 30
    """
    def __init__(self,  bymonthday=None, interval=1, tz=None):
        """
        mark every day in bymonthday; bymonthday can be an int or sequence

        Default is to tick every day of the month - bymonthday=range(1,32)
        """
        if bymonthday is None: bymonthday=range(1,32)
        o = rrulewrapper(DAILY, bymonthday=bymonthday,
                         interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 1

class HourLocator(RRuleLocator):
    """
    Make ticks on occurances of each hour
    """
    def __init__(self,  byhour=None, interval=1, tz=None):
        """
        mark every hour in byhour; byhour can be an int or sequence.
        Default is to tick every hour - byhour=range(24)

        interval is the interval between each iteration.  Eg, if
        interval=2, mark every second occurance
        """
        if byhour is None: byhour=range(24)
        rule = rrulewrapper(HOURLY, byhour=byhour, interval=interval,
                            byminute=0, bysecond=0)
        RRuleLocator.__init__(self, rule, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 1/24.

class MinuteLocator(RRuleLocator):
    """
    Make ticks on occurances of each minute
    """
    def __init__(self,  byminute=None, interval=1, tz=None):
        """
        mark every minute in byminute; byminute can be an int or
        sequence.  default is to tick every minute - byminute=range(60)

        interval is the interval between each iteration.  Eg, if
        interval=2, mark every second occurance

        """
        if byminute is None: byminute=range(60)
        rule = rrulewrapper(MINUTELY, byminute=byminute, interval=interval,
                            bysecond=0)
        RRuleLocator.__init__(self, rule, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 1./(24*60)

class SecondLocator(RRuleLocator):
    """
    Make ticks on occurances of each second
    """
    def __init__(self,  bysecond=None, interval=1, tz=None):
        """
        mark every second in bysecond; bysecond can be an int or
        sequence.  Default is to tick every second bysecond = range(60)

        interval is the interval between each iteration.  Eg, if
        interval=2, mark every second occurance

        """
        if bysecond is None: bysecond=range(60)
        rule = rrulewrapper(SECONDLY, bysecond=bysecond, interval=interval)
        RRuleLocator.__init__(self, rule, tz)

    def _get_unit(self):
        """
        return how many days a unit of the locator is; use for
        intelligent autoscaling
        """
        return 1./(24*60*60)



def _close_to_dt(d1, d2, epsilon=5):
    'assert that datetimes d1 and d2 are within epsilon microseconds'
    delta = d2-d1
    mus = abs(delta.days*MUSECONDS_PER_DAY + delta.seconds*1e6 +
              delta.microseconds)
    assert(mus<epsilon)

def _close_to_num(o1, o2, epsilon=5):
    'assert that float ordinals o1 and o2 are within epsilon microseconds'
    delta = abs((o2-o1)*MUSECONDS_PER_DAY)
    assert(delta<epsilon)

def epoch2num(e):
    """
    convert an epoch or sequence of epochs to the new date format,
    days since 0001
    """
    spd = 24.*3600.
    return 719163 + npy.asarray(e)/spd

def num2epoch(d):
    """
    convert days since 0001 to epoch.  d can be a number or sequence
    """
    spd = 24.*3600.
    return (npy.asarray(d)-719163)*spd

def mx2num(mxdates):
    """
    Convert mx datetime instance (or sequence of mx instances) to the
    new date format,
    """
    scalar = False
    if not cbook.iterable(mxdates):
        scalar = True
        mxdates = [mxdates]
    ret = epoch2num([m.ticks() for m in mxdates])
    if scalar: return ret[0]
    else: return ret


def date_ticker_factory(span, tz=None, numticks=5):
    """
    Create a date locator with numticks (approx) and a date formatter
    for span in days.  Return value is (locator, formatter)


    """

    if span==0: span = 1/24.

    minutes = span*24*60
    hours  = span*24
    days   = span
    weeks  = span/7.
    months = span/31. # approx
    years  = span/365.

    if years>numticks:
        locator = YearLocator(int(years/numticks), tz=tz)  # define
        fmt = '%Y'
    elif months>numticks:
        locator = MonthLocator(tz=tz)
        fmt = '%b %Y'
    elif weeks>numticks:
        locator = WeekdayLocator(tz=tz)
        fmt = '%a, %b %d'
    elif days>numticks:
        locator = DayLocator(interval=int(math.ceil(days/numticks)), tz=tz)
        fmt = '%b %d'
    elif hours>numticks:
        locator = HourLocator(interval=int(math.ceil(hours/numticks)), tz=tz)
        fmt = '%H:%M\n%b %d'
    elif minutes>numticks:
        locator = MinuteLocator(interval=int(math.ceil(minutes/numticks)), tz=tz)
        fmt = '%H:%M:%S'
    else:
        locator = MinuteLocator(tz=tz)
        fmt = '%H:%M:%S'


    formatter = DateFormatter(fmt, tz=tz)
    return locator, formatter


def seconds(s):
    'return seconds as days'
    return float(s)/SEC_PER_DAY

def minutes(m):
    'return minutes as days'
    return float(m)/MINUTES_PER_DAY

def hours(h):
    'return hours as days'
    return h/24.

def weeks(w):
    'return weeks as days'
    return w*7.


class DateConverter(units.ConversionInterface):

    def axisinfo(unit):
        'return the unit AxisInfo'
        if unit=='date':
            majloc = AutoDateLocator()
            majfmt = AutoDateFormatter(majloc)
            return units.AxisInfo(
                majloc = majloc,
                majfmt = majfmt,
                label='date',
                )
        else: return None
    axisinfo = staticmethod(axisinfo)

    def convert(value, unit):
        if units.ConversionInterface.is_numlike(value): return value
        return date2num(value)
    convert = staticmethod(convert)

    def default_units(x):
        'return the default unit for x or None'
        return 'date'
    default_units = staticmethod(default_units)


units.registry[datetime.date] = DateConverter()
units.registry[datetime.datetime] = DateConverter()



if __name__=='__main__':

    #tz = None
    tz = pytz.timezone('US/Pacific')
    #tz = UTC

    dt = datetime.datetime(1011, 10, 9, 13, 44, 22, 101010, tzinfo=tz)
    x = date2num(dt)
    _close_to_dt(dt, num2date(x, tz))

    #tz = _get_rc_timezone()


    d1 = datetime.datetime( 2000, 3, 1, tzinfo=tz)
    d2 = datetime.datetime( 2000, 3, 5, tzinfo=tz)

    #d1 = datetime.datetime( 2002, 1, 5, tzinfo=tz)
    #d2 = datetime.datetime( 2003, 12, 1, tzinfo=tz)
    delta = datetime.timedelta(hours=6)
    dates = drange(d1, d2, delta)

    #print 'orig', d1
    #print 'd2n and back', num2date(date2num(d1), tz)
    from _transforms import Value, Interval
    v1 = Value(date2num(d1))
    v2 = Value(date2num(d2))
    dlim = Interval(v1,v2)
    vlim = Interval(v1,v2)

    #locator = HourLocator(byhour=(3,15), tz=tz)
    #locator = MinuteLocator(byminute=(15,30,45), tz=tz)
    #locator = YearLocator(base=5, month=7, day=4, tz=tz)
    #locator = MonthLocator(bymonthday=15)
    locator = DayLocator(tz=tz)
    locator.set_data_interval(dlim)
    locator.set_view_interval(vlim)
    dmin, dmax = locator.autoscale()
    vlim.set_bounds(dmin, dmax)
    ticks =  locator()


    fmt = '%Y-%m-%d %H:%M:%S %Z'
    formatter = DateFormatter(fmt, tz)

    #for t in  ticks: print formatter(t)

    for t in dates: print formatter(t)



