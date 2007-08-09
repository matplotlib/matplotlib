'''tzinfo timezone information for Etc/GMT_plus_6.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_6(StaticTzInfo):
    '''Etc/GMT_plus_6 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_6'
    _utcoffset = timedelta(seconds=-21600)
    _tzname = 'GMT+6'

GMT_plus_6 = GMT_plus_6()

