'''tzinfo timezone information for Zulu.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Zulu(StaticTzInfo):
    '''Zulu timezone definition. See datetime.tzinfo for details'''
    zone = 'Zulu'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'UTC'

Zulu = Zulu()

