'''tzinfo timezone information for Africa/Djibouti.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Djibouti(DstTzInfo):
    '''Africa/Djibouti timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Djibouti'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,6,30,21,7,24),
        ]

    _transition_info = [
i(10380,0,'LMT'),
i(10800,0,'EAT'),
        ]

Djibouti = Djibouti()

