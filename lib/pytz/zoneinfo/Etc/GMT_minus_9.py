'''tzinfo timezone information for Etc/GMT_minus_9.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_9(StaticTzInfo):
    '''Etc/GMT_minus_9 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_9'
    _utcoffset = timedelta(seconds=32400)
    _tzname = 'GMT-9'

GMT_minus_9 = GMT_minus_9()

