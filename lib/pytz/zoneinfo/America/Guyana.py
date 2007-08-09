'''tzinfo timezone information for America/Guyana.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Guyana(DstTzInfo):
    '''America/Guyana timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Guyana'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1915,3,1,3,52,40),
d(1966,5,26,3,45,0),
d(1975,7,31,3,45,0),
d(1991,1,1,3,0,0),
        ]

    _transition_info = [
i(-13980,0,'LMT'),
i(-13500,0,'GBGT'),
i(-13500,0,'GYT'),
i(-10800,0,'GYT'),
i(-14400,0,'GYT'),
        ]

Guyana = Guyana()

