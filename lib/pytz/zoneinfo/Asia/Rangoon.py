'''tzinfo timezone information for Asia/Rangoon.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Rangoon(DstTzInfo):
    '''Asia/Rangoon timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Rangoon'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1919,12,31,17,35,24),
d(1942,4,30,17,30,0),
d(1945,5,2,15,0,0),
        ]

    _transition_info = [
i(23100,0,'RMT'),
i(23400,0,'BURT'),
i(32400,0,'JST'),
i(23400,0,'MMT'),
        ]

Rangoon = Rangoon()

