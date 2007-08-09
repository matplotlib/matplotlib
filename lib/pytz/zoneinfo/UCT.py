'''tzinfo timezone information for UCT.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class UCT(StaticTzInfo):
    '''UCT timezone definition. See datetime.tzinfo for details'''
    zone = 'UCT'
    _utcoffset = timedelta(seconds=0)
    _tzname = 'UCT'

UCT = UCT()

