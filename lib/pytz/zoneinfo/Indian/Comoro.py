'''tzinfo timezone information for Indian/Comoro.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Comoro(DstTzInfo):
    '''Indian/Comoro timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Comoro'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,6,30,21,6,56),
        ]

    _transition_info = [
i(10380,0,'LMT'),
i(10800,0,'EAT'),
        ]

Comoro = Comoro()

