'''tzinfo timezone information for Etc/GMT_minus_7.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_7(StaticTzInfo):
    '''Etc/GMT_minus_7 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_7'
    _utcoffset = timedelta(seconds=25200)
    _tzname = 'GMT-7'

GMT_minus_7 = GMT_minus_7()

