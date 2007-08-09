'''tzinfo timezone information for Etc/GMT_minus_8.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_8(StaticTzInfo):
    '''Etc/GMT_minus_8 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_8'
    _utcoffset = timedelta(seconds=28800)
    _tzname = 'GMT-8'

GMT_minus_8 = GMT_minus_8()

