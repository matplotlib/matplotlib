'''tzinfo timezone information for Pacific/Fakaofo.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Fakaofo(StaticTzInfo):
    '''Pacific/Fakaofo timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Fakaofo'
    _utcoffset = timedelta(seconds=-36000)
    _tzname = 'TKT'

Fakaofo = Fakaofo()

