"""

Matplotlib provides sophisticated date plotting capabilites, standing
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
to use a custom time zone, pass a matplotlib.pytz.timezone instance
with the tz keyword argument to num2date, plot_date, and any custom
date tickers or locators you create.  See http://pytz.sourceforge.net
for information on pytz and timezone handling.

dateutils https://moin.conectiva.com.br/DateUtil the code to handle
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
import sys

import matplotlib

try: import datetime
except ImportError:
    raise ValueError('matplotlib %s date handling requires python2.3' % matplotlib.__version__)

from cbook import iterable
from pytz import timezone
from numerix import arange
from ticker import Formatter, Locator, Base
from dateutil.rrule import rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,\
     MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY
from dateutil.relativedelta import relativedelta

UTC      = timezone('UTC')
Eastern  = timezone('US/Eastern')
Central  = timezone('US/Central')
Mountain = timezone('US/Mountain')
Pacific  = timezone('US/Pacific')
London   = timezone('Europe/London')
Paris    = timezone('Europe/Paris')
Berlin   = timezone('Europe/Berlin')
Moscow   = timezone('Europe/Moscow')

def _get_rc_timezone():
    s = matplotlib.rcParams['timezone']
    return timezone(s)


HOURS_PER_DAY = 24.
MINUTES_PER_DAY  = 60.*HOURS_PER_DAY
SECONDS_PER_DAY =  60.*MINUTES_PER_DAY
MUSECONDS_PER_DAY = 1e6*SECONDS_PER_DAY
SEC_PER_MIN = 60
SEC_PER_HOUR = 3600
SEC_PER_DAY = SEC_PER_HOUR * 24
SEC_PER_WEEK = SEC_PER_DAY * 7
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = MO, TU, WE, TH, FR, SA, SU
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)

__all__ = ['date2num', 'num2date', 'drange', 'HOURS_PER_DAY',
           'MINUTES_PER_DAY', 'SECONDS_PER_DAY', 'MUSECONDS_PER_DAY',
           'SEC_PER_MIN', 'SEC_PER_HOUR', 'SEC_PER_DAY',
           'SEC_PER_WEEK', 'MONDAY', 'TUESDAY', 'WEDNESDAY',
           'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 'MO', 'TU',
           'WE', 'TH', 'FR', 'SA', 'SU', 'WEEKDAYS', 'YEARLY',
           'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY',
           'SECONDLY', 'DateFormatter', 'rrulewrapper',
           'RRuleLocator', 'YearLocator', 'MonthLocator',
           'WeekdayLocator', 'DayLocator', 'HourLocator',
           'MinuteLocator', 'SecondLocator', 'timezone']

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
    
    base =  dt.toordinal()
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
    remainder = x - ix
    hour, remainder = divmod(24*remainder, 1)
    minute, remainder = divmod(60*remainder, 1)
    second, remainder = divmod(60*remainder, 1)
    microsecond = int(1e6*remainder)
    if microsecond<10: microsecond=0 # compensate for rounding errors
    dt = datetime.datetime(dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond, tzinfo=UTC).astimezone(tz)

    
    if microsecond>999990:  # compensate for rounding errors
        dt += datetime.timedelta(microseconds=1e6-microsecond)
        
    return dt

def date2num(d):
    """
    d is either a datetime instance or a sequence of datetimes

    return value is a floating point number (or sequence of floats)
    which gives number of days (fraction part represents hours,
    minutes, seconds) since 0001-01-01 00:00:00 UTC
    """
    if not iterable(d): return _to_ordinalf(d)
    else: return [_to_ordinalf(val) for val in d]


def num2date(x, tz=None):
    """
    x is a float value which gives number of days (fraction part
    represents hours, minutes, seconds) since 0001-01-01 00:00:00 UTC

    Return value is a datetime instance in timezone tz (default to
    rcparams TZ value)

    if x is a sequence, a sequence of datetimes will be returned
    """
    if tz is None: tz = _get_rc_timezone()
    if not iterable(x): return _from_ordinalf(x, tz)
    else: return [_from_ordinalf(val, tz) for val in x]
    
def drange(dstart, dend, delta):
    """
    Return a date range as float gregorian ordinals.  dstart and dend
    are datetime instances.  delta is a datetime.timedelta instance
    """
    step = delta.days + delta.seconds/SECONDS_PER_DAY + delta.microseconds/MUSECONDS_PER_DAY
    f1 = _to_ordinalf(dstart)
    f2 = _to_ordinalf(dend)
    return arange(f1, f2, step)



### date tickers and formatters ###



class DateFormatter(Formatter):
    """
    Tick location is seconds since the epoch.  Use a strftime format
    string
    """

    def __init__(self, fmt, tz=None):
        """
        fmt is an strftime format string; tz is the tzinfo instance
        """
        if tz is None: tz = _get_rc_timezone()
        self.fmt = fmt
        self.tz = tz

    def __call__(self, x, pos=0):
        dt = num2date(x, self.tz)
        return dt.strftime(self.fmt)

    def set_tzinfo(self, tz):
        self.tz = tz

class IndexDateFormatter(Formatter):
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
        ind = int(x)
        if ind>=len(self.t) or ind<=0: return ''
        
        dt = num2date(self.t[ind], self.tz)
        return dt.strftime(self.fmt)

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

class DateLocator(Locator):
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


class RRuleLocator(DateLocator):
    # use the dateutil rrule instance 
        
    def __init__(self, o, tz=None):
        DateLocator.__init__(self, tz)
        self.rule = o

    def __call__(self):
        self.verify_intervals()

        dmin, dmax = self.viewlim_to_dt()
        delta = relativedelta(dmax, dmin)
        self.rule.set(dtstart=dmin-delta, until=dmax+delta)
        dates = self.rule.between(dmin, dmax, True)        
        return date2num(dates)

    def autoscale(self):
        """
        Set the view limits to include the data range
        """        
        self.verify_intervals()        
        dmin, dmax = self.datalim_to_dt()
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
        self.base = Base(base)
        self.replaced = { 'month'  : month,
                          'day'    : day,
                          'hour'   : 0,
                          'minute' : 0,
                          'second' : 0,
                          'tzinfo' : tz
                          }


    def __call__(self):
        self.verify_intervals()

        dmin, dmax = self.viewlim_to_dt()
        ymin = self.base.le(dmin.year)
        ymax = self.base.ge(dmax.year)    #print 'DMIN', num2date(dmin, tz).strftime(fmt)
    #print 'DMAX', num2date(dmax, tz).strftime(fmt)


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
        o = rrulewrapper(DAILY, byweekday=byweekday, interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)


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
        o = rrulewrapper(DAILY, bymonthday=bymonthday, interval=interval, **self.hms0d)
        RRuleLocator.__init__(self, o, tz)

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



def _close_to_dt(d1, d2, epsilon=5):
   'assert that datetimes d1 and d2 are within epsilon microseconds'
   delta = d2-d1
   mus = abs(delta.days*MUSECONDS_PER_DAY + delta.seconds*1e6 + delta.microseconds)
   assert(mus<epsilon)

def _close_to_num(o1, o2, epsilon=5):
   'assert that float ordinals o1 and o2 are within epsilon microseconds'
   delta = abs((o2-o1)*MUSECONDS_PER_DAY)
   assert(delta<epsilon)

if __name__=='__main__':
    
    #tz = None
    tz = Pacific
    #tz = UTC

    dt = datetime.datetime(1011, 10, 9, 13, 44, 22, 101010, tzinfo=tz)
    x = date2num(dt)
    _close_to_dt(dt, num2date(x, tz))

    #tz = _get_rc_timezone()


    d1 = datetime.datetime( 2000, 3, 1, tzinfo=tz)
    d2 = datetime.datetime( 2000, 3, 5, tzinfo=tz)

    #d1 = datetime.datetime( 2002, 1, 5, tzinfo=tz)
    #d2 = datetime.datetime( 2003, 12, 1, tzinfo=tz)
    print d1, d2
    delta = datetime.timedelta(hours=6)
    dates = drange(d1, d2, delta)
    print 'len dates', len(dates)

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
    print 'DMIN', formatter(dmin)
    print 'DMAX', formatter(dmax)

    #for t in  ticks: print formatter(t)

    for t in dates: print formatter(t)
    

