'''tzinfo timezone information for Africa/Blantyre.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Blantyre(DstTzInfo):
    '''Africa/Blantyre timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Blantyre'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,21,40,0),
        ]

    _transition_info = [
i(8400,0,'LMT'),
i(7200,0,'CAT'),
        ]

Blantyre = Blantyre()

