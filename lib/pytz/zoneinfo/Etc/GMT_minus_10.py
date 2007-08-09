'''tzinfo timezone information for Etc/GMT_minus_10.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_10(StaticTzInfo):
    '''Etc/GMT_minus_10 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_10'
    _utcoffset = timedelta(seconds=36000)
    _tzname = 'GMT-10'

GMT_minus_10 = GMT_minus_10()

