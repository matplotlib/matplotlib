'''tzinfo timezone information for Asia/Pyongyang.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Pyongyang(DstTzInfo):
    '''Asia/Pyongyang timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Pyongyang'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1904,11,30,15,30,0),
d(1927,12,31,15,0,0),
d(1931,12,31,15,30,0),
d(1954,3,20,15,0,0),
d(1961,8,9,16,0,0),
        ]

    _transition_info = [
i(30600,0,'KST'),
i(32400,0,'KST'),
i(30600,0,'KST'),
i(32400,0,'KST'),
i(28800,0,'KST'),
i(32400,0,'KST'),
        ]

Pyongyang = Pyongyang()

