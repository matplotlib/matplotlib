'''tzinfo timezone information for Pacific/Johnston.'''
from pytz.tzinfo import StaticTzInfo
from pytz.tzinfo import memorized_timedelta as timedelta

class Johnston(StaticTzInfo):
    '''Pacific/Johnston timezone definition. See datetime.tzinfo for details'''
    zone = 'Pacific/Johnston'
    _utcoffset = timedelta(seconds=-36000)
    _tzname = 'HST'

Johnston = Johnston()

