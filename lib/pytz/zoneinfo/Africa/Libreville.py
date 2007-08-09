'''tzinfo timezone information for Africa/Libreville.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Libreville(DstTzInfo):
    '''Africa/Libreville timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Libreville'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,23,22,12),
        ]

    _transition_info = [
i(2280,0,'LMT'),
i(3600,0,'WAT'),
        ]

Libreville = Libreville()

