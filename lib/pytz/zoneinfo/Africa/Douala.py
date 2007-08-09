'''tzinfo timezone information for Africa/Douala.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Douala(DstTzInfo):
    '''Africa/Douala timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Douala'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,23,21,12),
        ]

    _transition_info = [
i(2340,0,'LMT'),
i(3600,0,'WAT'),
        ]

Douala = Douala()

