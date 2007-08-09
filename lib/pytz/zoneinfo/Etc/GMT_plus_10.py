'''tzinfo timezone information for Etc/GMT_plus_10.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class GMT_plus_10(StaticTzInfo):
    '''Etc/GMT_plus_10 timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/GMT_plus_10'
    _utcoffset = timedelta(seconds=-36000)
    _tzname = 'GMT+10'

GMT_plus_10 = GMT_plus_10()

