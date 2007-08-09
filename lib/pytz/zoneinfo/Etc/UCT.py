'''tzinfo timezone information for Etc/UCT.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class UCT(StaticTzInfo):
    '''Etc/UCT timezone definition. See datetime.tzinfo for details'''
    zone = 'Etc/UCT'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'UCT'

UCT = UCT()

