'''tzinfo timezone information for Pacific/Midway.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class Midway(DstTzInfo):
    '''Pacific/Midway timezone definition. See datetime.tzinfo for details'''

    zone = 'Pacific/Midway'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1956,6,3,11,0,0),
d(1956,9,2,10,0,0),
d(1967,4,1,11,0,0),
d(1983,11,30,11,0,0),
        ]

    _transition_info = [
i(-39600,0,'NST'),
i(-36000,3600,'NDT'),
i(-39600,0,'NST'),
i(-39600,0,'BST'),
i(-39600,0,'SST'),
        ]

Midway = Midway()

