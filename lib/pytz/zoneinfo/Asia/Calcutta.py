'''tzinfo timezone information for Asia/Calcutta.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Calcutta(DstTzInfo):
    '''Asia/Calcutta timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Calcutta'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1941,9,30,18,6,40),
d(1942,5,14,17,30,0),
d(1942,8,31,18,30,0),
d(1945,10,14,17,30,0),
        ]

    _transition_info = [
i(21180,0,'HMT'),
i(23400,0,'BURT'),
i(19800,0,'IST'),
i(23400,3600,'IST'),
i(19800,0,'IST'),
        ]

Calcutta = Calcutta()

