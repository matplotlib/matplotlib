'''tzinfo timezone information for Asia/Tokyo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Tokyo(DstTzInfo):
    '''Asia/Tokyo timezone definition. See datetime.tzinfo for details'''

    _zone = 'Asia/Tokyo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1937,12,31,15,0,0),
        ]

    _transition_info = [
i(32400,0,'CJT'),
i(32400,0,'JST'),
        ]

Tokyo = Tokyo()

