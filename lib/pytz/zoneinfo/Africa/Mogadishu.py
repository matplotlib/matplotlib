'''tzinfo timezone information for Africa/Mogadishu.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Mogadishu(DstTzInfo):
    '''Africa/Mogadishu timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Mogadishu'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1930,12,31,21,0,0),
d(1956,12,31,21,30,0),
        ]

    _transition_info = [
i(10800,0,'EAT'),
i(9000,0,'BEAT'),
i(10800,0,'EAT'),
        ]

Mogadishu = Mogadishu()

