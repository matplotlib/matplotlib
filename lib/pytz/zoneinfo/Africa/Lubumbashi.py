'''tzinfo timezone information for Africa/Lubumbashi.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Lubumbashi(StaticTzInfo):
    '''Africa/Lubumbashi timezone definition. See datetime.tzinfo for details'''
    zone = 'Africa/Lubumbashi'
    _utcoffset = timedelta(seconds=7200)
    _tzname = 'CAT'

Lubumbashi = Lubumbashi()

