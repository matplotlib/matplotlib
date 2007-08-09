'''tzinfo timezone information for Atlantic/Cape_Verde.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Cape_Verde(DstTzInfo):
    '''Atlantic/Cape_Verde timezone definition. See datetime.tzinfo for details'''

    zone = 'Atlantic/Cape_Verde'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1907,1,1,1,34,4),
d(1942,9,1,2,0,0),
d(1945,10,15,1,0,0),
d(1975,11,25,4,0,0),
        ]

    _transition_info = [
i(-5640,0,'LMT'),
i(-7200,0,'CVT'),
i(-3600,3600,'CVST'),
i(-7200,0,'CVT'),
i(-3600,0,'CVT'),
        ]

Cape_Verde = Cape_Verde()

