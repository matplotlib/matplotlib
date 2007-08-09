'''tzinfo timezone information for Asia/Dubai.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dubai(DstTzInfo):
    '''Asia/Dubai timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Dubai'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,20,18,48),
        ]

    _transition_info = [
i(13260,0,'LMT'),
i(14400,0,'GST'),
        ]

Dubai = Dubai()

