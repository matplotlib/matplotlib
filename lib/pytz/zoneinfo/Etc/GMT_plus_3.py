'''tzinfo timezone information for Etc/GMT_plus_3.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_3(StaticTzInfo):
    '''Etc/GMT_plus_3 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_3'
    _utcoffset = timedelta(seconds=-10800)
    _tzname = 'GMT+3'

GMT_plus_3 = GMT_plus_3()

