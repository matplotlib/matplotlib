'''tzinfo timezone information for Africa/El_Aaiun.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class El_Aaiun(DstTzInfo):
    '''Africa/El_Aaiun timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/El_Aaiun'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1934,1,1,0,52,48),
d(1976,4,14,1,0,0),
        ]

    _transition_info = [
i(-3180,0,'LMT'),
i(-3600,0,'WAT'),
i(0,0,'WET'),
        ]

El_Aaiun = El_Aaiun()

