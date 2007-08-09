'''tzinfo timezone information for HST.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class HST(StaticTzInfo):
    '''HST timezone definition. See datetime.tzinfo for details'''
    zone = 'HST'
    _utcoffset = timedelta(seconds=-36000)
    _tzname = 'HST'

HST = HST()

