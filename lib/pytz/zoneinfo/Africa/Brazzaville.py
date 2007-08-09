'''tzinfo timezone information for Africa/Brazzaville.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Brazzaville(DstTzInfo):
    '''Africa/Brazzaville timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Brazzaville'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,22,58,52),
        ]

    _transition_info = [
i(3660,0,'LMT'),
i(3600,0,'WAT'),
        ]

Brazzaville = Brazzaville()

