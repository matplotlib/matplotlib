'''tzinfo timezone information for EST.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class EST(StaticTzInfo):
    '''EST timezone definition. See datetime.tzinfo for details'''
    zone = 'EST'
    _utcoffset = timedelta(seconds=-18000)
    _tzname = 'EST'

EST = EST()

