'''tzinfo timezone information for Etc/GMT.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT(StaticTzInfo):
    '''Etc/GMT timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'GMT'

GMT = GMT()

