'''tzinfo timezone information for Asia/Karachi.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Karachi(DstTzInfo):
    '''Asia/Karachi timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Karachi'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1906,12,31,19,31,48),
d(1942,8,31,18,30,0),
d(1945,10,14,17,30,0),
d(1951,9,29,18,30,0),
d(1971,3,25,19,0,0),
d(2002,4,6,19,1,0),
d(2002,10,5,18,1,0),
        ]

    _transition_info = [
i(16080,0,'LMT'),
i(19800,0,'IST'),
i(23400,3600,'IST'),
i(19800,0,'IST'),
i(18000,0,'KART'),
i(18000,0,'PKT'),
i(21600,3600,'PKST'),
i(18000,0,'PKT'),
        ]

Karachi = Karachi()

