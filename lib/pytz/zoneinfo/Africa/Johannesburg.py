'''tzinfo timezone information for Africa/Johannesburg.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Johannesburg(DstTzInfo):
    '''Africa/Johannesburg timezone definition. See datetime.tzinfo for details'''

    zone = 'Africa/Johannesburg'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1903,2,28,22,30,0),
d(1942,9,20,0,0,0),
d(1943,3,20,23,0,0),
d(1943,9,19,0,0,0),
d(1944,3,18,23,0,0),
        ]

    _transition_info = [
i(5400,0,'SAST'),
i(7200,0,'SAST'),
i(10800,3600,'SAST'),
i(7200,0,'SAST'),
i(10800,3600,'SAST'),
i(7200,0,'SAST'),
        ]

Johannesburg = Johannesburg()

