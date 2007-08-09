'''tzinfo timezone information for Africa/Lagos.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Lagos(DstTzInfo):
    '''Africa/Lagos timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Lagos'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,8,31,23,46,24),
        ]

    _transition_info = [
i(840,0,'LMT'),
i(3600,0,'WAT'),
        ]

Lagos = Lagos()

