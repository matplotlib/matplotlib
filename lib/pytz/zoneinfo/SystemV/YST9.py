'''tzinfo timezone information for SystemV/YST9.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class YST9(DstTzInfo):
    '''SystemV/YST9 timezone definition. See datetime.tzinfo for details'''

    _zone = 'SystemV/YST9'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,10,1,8,59,48),
        ]

    _transition_info = [
i(-32400,0,'LMT'),
i(-32400,0,'GAMT'),
        ]

YST9 = YST9()

