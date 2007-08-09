'''tzinfo timezone information for Africa/Lome.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Lome(StaticTzInfo):
    '''Africa/Lome timezone definition. See datetime.tzinfo for details'''
    zone = 'Africa/Lome'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'GMT'

Lome = Lome()

