'''tzinfo timezone information for America/Phoenix.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Phoenix(DstTzInfo):
    '''America/Phoenix timezone definition. See datetime.tzinfo for details'''

    zone = 'America/Phoenix'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1918,3,31,9,0,0),
d(1918,10,27,8,0,0),
d(1919,3,30,9,0,0),
d(1919,10,26,8,0,0),
d(1942,2,9,9,0,0),
d(1944,1,1,6,1,0),
d(1944,4,1,7,1,0),
d(1944,10,1,6,1,0),
d(1967,4,30,9,0,0),
d(1967,10,29,8,0,0),
        ]

    _transition_info = [
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
i(-21600,3600,'MWT'),
i(-25200,0,'MST'),
i(-21600,3600,'MWT'),
i(-25200,0,'MST'),
i(-21600,3600,'MDT'),
i(-25200,0,'MST'),
        ]

Phoenix = Phoenix()

