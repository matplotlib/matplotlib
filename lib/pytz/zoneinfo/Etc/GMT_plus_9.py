'''tzinfo timezone information for Etc/GMT_plus_9.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_9(StaticTzInfo):
    '''Etc/GMT_plus_9 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_9'
    _utcoffset = timedelta(seconds=-32400)
    _tzname = 'GMT+9'

GMT_plus_9 = GMT_plus_9()

