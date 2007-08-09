'''tzinfo timezone information for Asia/Dili.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Dili(DstTzInfo):
    '''Asia/Dili timezone definition. See datetime.tzinfo for details'''

    zone = 'Asia/Dili'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1911,12,31,15,37,40),
d(1942,2,21,15,0,0),
d(1945,7,31,15,0,0),
d(1976,5,2,15,0,0),
d(2000,9,16,16,0,0),
        ]

    _transition_info = [
i(30120,0,'LMT'),
i(28800,0,'TLT'),
i(32400,0,'JST'),
i(32400,0,'TLT'),
i(28800,0,'CIT'),
i(32400,0,'TLT'),
        ]

Dili = Dili()

