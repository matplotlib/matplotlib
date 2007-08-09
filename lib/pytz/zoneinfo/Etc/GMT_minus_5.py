'''tzinfo timezone information for Etc/GMT_minus_5.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_5(StaticTzInfo):
    '''Etc/GMT_minus_5 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_5'
    _utcoffset = timedelta(seconds=18000)
    _tzname = 'GMT-5'

GMT_minus_5 = GMT_minus_5()

