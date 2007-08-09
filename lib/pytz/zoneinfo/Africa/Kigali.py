'''tzinfo timezone information for Africa/Kigali.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Kigali(DstTzInfo):
    '''Africa/Kigali timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Kigali'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1935,5,31,21,59,44),
        ]

    _transition_info = [
i(7200,0,'LMT'),
i(7200,0,'CAT'),
        ]

Kigali = Kigali()

