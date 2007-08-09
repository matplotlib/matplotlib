'''tzinfo timezone information for Africa/Banjul.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Banjul(DstTzInfo):
    '''Africa/Banjul timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Banjul'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1912,1,1,1,6,36),
d(1935,1,1,1,6,36),
d(1964,1,1,1,0,0),
        ]

    _transition_info = [
i(-4020,0,'LMT'),
i(-4020,0,'BMT'),
i(-3600,0,'WAT'),
i(0,0,'GMT'),
        ]

Banjul = Banjul()

