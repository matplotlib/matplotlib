'''tzinfo timezone information for Etc/GMT_minus_13.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_13(StaticTzInfo):
    '''Etc/GMT_minus_13 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_13'
    _utcoffset = timedelta(seconds=46800)
    _tzname = 'GMT-13'

GMT_minus_13 = GMT_minus_13()

