'''tzinfo timezone information for Africa/Dakar.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dakar(DstTzInfo):
    '''Africa/Dakar timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Dakar'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,1,9,44),
d(1941,6,1,1,0,0),
        ]

    _transition_info = [
i(-4200,0,'LMT'),
i(-3600,0,'WAT'),
i(0,0,'GMT'),
        ]

Dakar = Dakar()

