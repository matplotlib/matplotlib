'''tzinfo timezone information for SystemV/PST8.'''
from pytz.tzinfo import DstTzInfo
from pytz.tzinfo import memorized_datetime as d
from pytz.tzinfo import memorized_ttinfo as i

class PST8(DstTzInfo):
    '''SystemV/PST8 timezone definition. See datetime.tzinfo for details'''

    _zone = 'SystemV/PST8'

    _utc_transition_times = [
d(1,1,1,0,0,0),
d(1998,4,27,8,30,0),
        ]

    _transition_info = [
i(-30600,0,'PNT'),
i(-28800,0,'PST'),
        ]

PST8 = PST8()

