'''tzinfo timezone information for Asia/Muscat.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Muscat(DstTzInfo):
    '''Asia/Muscat timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Muscat'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,20,5,40),
        ]

    _transition_info = [
i(14040,0,'LMT'),
i(14400,0,'GST'),
        ]

Muscat = Muscat()

