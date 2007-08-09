'''tzinfo timezone information for Australia/Brisbane.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Brisbane(DstTzInfo):
    '''Australia/Brisbane timezone definition. See datetime.tzinfo for details'''

    zone = 'Australia/Brisbane'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1916,12,31,14,1,0),
d(1917,3,24,15,0,0),
d(1941,12,31,16,0,0),
d(1942,3,28,15,0,0),
d(1942,9,26,16,0,0),
d(1943,3,27,15,0,0),
d(1943,10,2,16,0,0),
d(1944,3,25,15,0,0),
d(1971,10,30,16,0,0),
d(1972,2,26,16,0,0),
d(1989,10,28,16,0,0),
d(1990,3,3,16,0,0),
d(1990,10,27,16,0,0),
d(1991,3,2,16,0,0),
d(1991,10,26,16,0,0),
d(1992,2,29,16,0,0),
        ]

    _transition_info = [
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
i(39600,3600,'EST'),
i(36000,0,'EST'),
        ]

Brisbane = Brisbane()

