'''tzinfo timezone information for America/Swift_Current.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Swift_Current(DstTzInfo):
    '''America/Swift_Current timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Swift_Current'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1905,9,1,7,11,20),
d(1918,4,14,9,0,0),
d(1918,10,31,8,0,0),
d(1942,2,9,9,0,0),
d(1945,8,14,23,0,0),
d(1945,9,30,8,0,0),
d(1946,4,28,9,0,0),
d(1946,10,13,8,0,0),
d(1947,4,27,9,0,0),
d(1947,9,28,8,0,0),
d(1948,4,25,9,0,0),
d(1948,9,26,8,0,0),
d(1949,4,24,9,0,0),
d(1949,9,25,8,0,0),
d(1957,4,28,9,0,0),
d(1957,10,27,8,0,0),
d(1959,4,26,9,0,0),
d(1959,10,25,8,0,0),
d(1960,4,24,9,0,0),
d(1960,9,25,8,0,0),
d(1961,4,30,9,0,0),
d(1961,9,24,8,0,0),
d(1972,4,30,9,0,0),
        ]

    _transition_info = [
i(-25860,0,'LMT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MWT'),
i(-21600,3600,'MPT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,0,'CST'),
        ]

Swift_Current = Swift_Current()

