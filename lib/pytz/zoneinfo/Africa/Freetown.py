'''tzinfo timezone information for Africa/Freetown.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Freetown(DstTzInfo):
    '''Africa/Freetown timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Freetown'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1913,6,1,0,53,0),
d(1935,6,1,1,0,0),
d(1935,10,1,0,20,0),
d(1936,6,1,1,0,0),
d(1936,10,1,0,20,0),
d(1937,6,1,1,0,0),
d(1937,10,1,0,20,0),
d(1938,6,1,1,0,0),
d(1938,10,1,0,20,0),
d(1939,6,1,1,0,0),
d(1939,10,1,0,20,0),
d(1940,6,1,1,0,0),
d(1940,10,1,0,20,0),
d(1941,6,1,1,0,0),
d(1941,10,1,0,20,0),
d(1942,6,1,1,0,0),
d(1942,10,1,0,20,0),
d(1957,1,1,1,0,0),
d(1957,6,1,0,0,0),
d(1957,8,31,23,0,0),
d(1958,6,1,0,0,0),
d(1958,8,31,23,0,0),
d(1959,6,1,0,0,0),
d(1959,8,31,23,0,0),
d(1960,6,1,0,0,0),
d(1960,8,31,23,0,0),
d(1961,6,1,0,0,0),
d(1961,8,31,23,0,0),
d(1962,6,1,0,0,0),
d(1962,8,31,23,0,0),
        ]

    _transition_info = [
i(-3180,0,'FMT'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(-1200,2400,'SLST'),
i(-3600,0,'WAT'),
i(0,0,'WAT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
i(3600,3600,'SLST'),
i(0,0,'GMT'),
        ]

Freetown = Freetown()

