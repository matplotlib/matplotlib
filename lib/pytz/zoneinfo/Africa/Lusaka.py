'''tzinfo timezone information for Africa/Lusaka.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Lusaka(DstTzInfo):
    '''Africa/Lusaka timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Lusaka'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,22,6,52),
        ]

    _transition_info = [
i(6780,0,'LMT'),
i(7200,0,'CAT'),
        ]

Lusaka = Lusaka()

