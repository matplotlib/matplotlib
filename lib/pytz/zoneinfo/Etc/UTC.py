'''tzinfo timezone information for Etc/UTC.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class UTC(StaticTzInfo):
    '''Etc/UTC timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/UTC'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'UTC'

UTC = UTC()

