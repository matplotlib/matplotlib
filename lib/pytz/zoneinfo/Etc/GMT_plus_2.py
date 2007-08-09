'''tzinfo timezone information for Etc/GMT_plus_2.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_2(StaticTzInfo):
    '''Etc/GMT_plus_2 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_2'
    _utcoffset = timedelta(seconds=-7200)
    _tzname = 'GMT+2'

GMT_plus_2 = GMT_plus_2()

