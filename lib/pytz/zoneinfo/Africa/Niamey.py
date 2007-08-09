'''tzinfo timezone information for Africa/Niamey.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Niamey(DstTzInfo):
    '''Africa/Niamey timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Niamey'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,23,51,32),
d(1934,2,26,1,0,0),
d(1960,1,1,0,0,0),
        ]

    _transition_info = [
i(480,0,'LMT'),
i(-3600,0,'WAT'),
i(0,0,'GMT'),
i(3600,0,'WAT'),
        ]

Niamey = Niamey()

