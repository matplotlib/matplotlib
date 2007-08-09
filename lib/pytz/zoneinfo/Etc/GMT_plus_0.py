'''tzinfo timezone information for Etc/GMT_plus_0.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_0(StaticTzInfo):
    '''Etc/GMT_plus_0 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_0'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'GMT'

GMT_plus_0 = GMT_plus_0()

