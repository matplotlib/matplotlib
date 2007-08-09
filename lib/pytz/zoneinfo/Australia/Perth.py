'''tzinfo timezone information for Australia/Perth.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Perth(DstTzInfo):
    '''Australia/Perth timezone definition. See datetime.tzinfo for details'''

    zone = 'Australia/Perth'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1916,12,31,16,1,0),
d(1917,3,24,17,0,0),
d(1941,12,31,18,0,0),
d(1942,3,28,17,0,0),
d(1942,9,26,18,0,0),
d(1943,3,27,17,0,0),
d(1974,10,26,18,0,0),
d(1975,3,1,18,0,0),
d(1983,10,29,18,0,0),
d(1984,3,3,18,0,0),
d(1991,11,16,18,0,0),
d(1992,2,29,18,0,0),
d(2006,12,2,18,0,0),
d(2007,3,24,18,0,0),
d(2007,10,27,18,0,0),
d(2008,3,29,18,0,0),
d(2008,10,25,18,0,0),
d(2009,3,28,18,0,0),
        ]

    _transition_info = [
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
i(32400,3600,'WST'),
i(28800,0,'WST'),
        ]

Perth = Perth()

