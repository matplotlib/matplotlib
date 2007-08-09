'''tzinfo timezone information for Indian/Mahe.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mahe(DstTzInfo):
    '''Indian/Mahe timezone definition. See datetime.tzinfo for details'''

    zone = 'Indian/Mahe'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1906,5,31,20,18,12),
        ]

    _transition_info = [
i(13320,0,'LMT'),
i(14400,0,'SCT'),
        ]

Mahe = Mahe()

