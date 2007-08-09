'''tzinfo timezone information for Indian/Reunion.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Reunion(DstTzInfo):
    '''Indian/Reunion timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Reunion'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,5,31,20,18,8),
        ]

    _transition_info = [
i(13320,0,'LMT'),
i(14400,0,'RET'),
        ]

Reunion = Reunion()

