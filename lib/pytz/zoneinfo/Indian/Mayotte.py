'''tzinfo timezone information for Indian/Mayotte.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mayotte(DstTzInfo):
    '''Indian/Mayotte timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Mayotte'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,6,30,20,59,4),
        ]

    _transition_info = [
i(10860,0,'LMT'),
i(10800,0,'EAT'),
        ]

Mayotte = Mayotte()

