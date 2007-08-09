'''tzinfo timezone information for MST.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class MST(StaticTzInfo):
    '''MST timezone definition. See datetime.tzinfo for details'''
    zone = 'MST'
    _utcoffset = timedelta(seconds=-25200)
    _tzname = 'MST'

MST = MST()

