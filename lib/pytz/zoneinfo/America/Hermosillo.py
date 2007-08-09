'''tzinfo timezone information for America/Hermosillo.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Hermosillo(DstTzInfo):
    '''America/Hermosillo timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Hermosillo'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1922,1,1,7,0,0),
d(1927,6,11,6,0,0),
d(1930,11,15,6,0,0),
d(1931,5,2,6,0,0),
d(1931,10,1,6,0,0),
d(1932,4,1,7,0,0),
d(1942,4,24,6,0,0),
d(1949,1,14,7,0,0),
d(1970,1,1,8,0,0),
d(1996,4,7,9,0,0),
d(1996,10,27,8,0,0),
d(1997,4,6,9,0,0),
d(1997,10,26,8,0,0),
d(1998,4,5,9,0,0),
d(1998,10,25,8,0,0),
        ]

    _transition_info = [
i(-26640,0,'LMT'),
i(-25200,0,'MST'),
i(-21600,0,'CST'),
i(-25200,0,'MST'),
i(-21600,0,'CST'),
i(-25200,0,'MST'),
i(-21600,0,'CST'),
i(-25200,0,'MST'),
i(-28800,0,'PST'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
        ]

Hermosillo = Hermosillo()

