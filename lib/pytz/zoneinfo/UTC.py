'''tzinfo timezone information for UTC.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class UTC(StaticTzInfo):
    '''UTC timezone definition. See datetime.tzinfo for details'''
    _zone = 'UTC'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'UTC'

UTC = UTC()

