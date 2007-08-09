'''tzinfo timezone information for Etc/GMT_minus_2.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_2(StaticTzInfo):
    '''Etc/GMT_minus_2 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_2'
    _utcoffset = timedelta(seconds=7200)
    _tzname = 'GMT-2'

GMT_minus_2 = GMT_minus_2()

