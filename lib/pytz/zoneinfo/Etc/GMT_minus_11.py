'''tzinfo timezone information for Etc/GMT_minus_11.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_minus_11(StaticTzInfo):
    '''Etc/GMT_minus_11 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_minus_11'
    _utcoffset = timedelta(seconds=39600)
    _tzname = 'GMT-11'

GMT_minus_11 = GMT_minus_11()

