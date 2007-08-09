'''tzinfo timezone information for Etc/GMT_plus_12.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_12(StaticTzInfo):
    '''Etc/GMT_plus_12 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_12'
    _utcoffset = timedelta(seconds=-43200)
    _tzname = 'GMT+12'

GMT_plus_12 = GMT_plus_12()

